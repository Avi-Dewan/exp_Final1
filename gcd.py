import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import SGD 
from torch.autograd import Variable

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.sinkhorn_knopp import SinkhornKnopp
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, softBCE, str2bool, PairEnum, BCE, myBCE, softBCE_F, softBCE_N
from utils import ramps 
from utils.simCLR_loss import SimCLR_Loss

from models.resnet import ResNet, BasicBlock 
from models.preModel import ProjectionHead
from models.resnetMultiHead import ResNetMultiHead

from modules.module import feat2prob, target_distribution 

from data.cifarloader import CIFAR10Loader

from tqdm import tqdm

import numpy as np
import warnings
import random
import os
import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

def init_prob_kmeans(model, eval_loader, args):
    '''
    Initilize Cluster centers.
    Calculate initial acc, nmi, ari.
    Calculate initial probability of target images
    '''
    torch.manual_seed(1)

    model = model.to(device)
    # cluster parameter initiate
    model.eval()

    targets = np.zeros(len(eval_loader.dataset)) # labels storage
    extracted_features = np.zeros((len(eval_loader.dataset), 512)) # features storage

    for _, (x, label, idx) in enumerate(eval_loader):
        x = x.to(device)

        extracted_feat, _ = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)

        idx = idx.data.cpu().numpy() # get the index
        extracted_features[idx, :] = extracted_feat.data.cpu().numpy()  # store the feature
        targets[idx] = label.data.cpu().numpy() # store the label

    # pca = PCA(n_components=args.n_unlabeled_classes)
    pca = PCA(n_components=20) # PCA for dimensionality reduction PCA: 512 -> 20

    extracted_features = pca.fit_transform(extracted_features) # fit the PCA model and transform the features
    kmeans = KMeans(n_clusters=args.n_unlabeled_classes, n_init=20)  # KMeans clustering
    y_pred = kmeans.fit_predict(extracted_features) # predict the cluster

    # evaluate clustering performance
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    # Find the probability distribution by calculating distance from the center
    probs = feat2prob(torch.from_numpy(extracted_features), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs 


def warmup_train(model, train_loader, eva_loader, args):
    '''
    Warmup Training:
    Anneal the probability distribution to target distribution.
    After warmup , update the target distribution
    '''
    optimizer = SGD(model.parameters(), lr=args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(args.warmup_epochs):
        loss_record = AverageMeter()

        model.train()

        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)

            _, _, z_unlabeled = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            prob = feat2prob(z_unlabeled, model.center) # get the probability distribution by calculating distance from the center
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        
    args.p_targets = target_distribution(probs)  # update the target distribution


def PI_CL_softBCE_train(model, unlabeled_train_loader, unlabeled_eval_loader, args):
    """
    Training with:
    - KL on sharpened targets
    - MSE consistency loss
    - SimCLR contrastive loss
    - Pairwise BCE loss (ranking-based)
    """
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()
    
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)

        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, labeled_pred, z_unlabeled = model(x)
            extracted_feat_bar, labeled_pred_bar, z_unlabeled_bar = model(x_bar)

            prob = feat2prob(z_unlabeled, model.center)
            prob_bar = feat2prob(z_unlabeled_bar, model.center)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # contrastive loss using internal projector
            z_i = model.projector_CL(extracted_feat)
            z_j = model.projector_CL(extracted_feat_bar)
            contrastive_loss = simCLR_loss(z_i, z_j)

            # pairwise BCE label via ranking
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            loss = sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model
    torch.save({'state_dict': model.state_dict(), 'center': model.center}, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="Accuracy")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')


def CE_PI_CL_softBCE_train(model, 
                           labeled_train_loader,
                           labeled_eval_loader,
                           unlabeled_train_loader,
                           unlabeled_eval_loader,
                           args):
    """
    Training with:
    - Cross-entropy loss for labeled data
    - KL on sharpened targets
    - MSE consistency loss
    - SimCLR contrastive loss
    - Pairwise BCE loss (ranking-based)
    """
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()
    criterion_ce = nn.CrossEntropyLoss()  # Cross-entropy loss for labeled data
    
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)

        # Create iterators for both labeled and unlabeled data
        labeled_iter = iter(labeled_train_loader)
        unlabeled_iter = iter(unlabeled_train_loader)
        
        # Determine the number of batches (use the smaller of the two)
        num_batches = min(len(labeled_train_loader), len(unlabeled_train_loader))

        for batch_idx in range(num_batches):
            # Get labeled batch
            try:
                (x_labeled, x_labeled_bar), label_labeled, idx_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_train_loader)
                (x_labeled, x_labeled_bar), label_labeled, idx_labeled = next(labeled_iter)
            
            # Get unlabeled batch
            try:
                (x_unlabeled, x_unlabeled_bar), label_unlabeled, idx_unlabeled = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_train_loader)
                (x_unlabeled, x_unlabeled_bar), label_unlabeled, idx_unlabeled = next(unlabeled_iter)

            # Move to device
            x_labeled, x_labeled_bar = x_labeled.to(device), x_labeled_bar.to(device)
            label_labeled = label_labeled.to(device)
            x_unlabeled, x_unlabeled_bar = x_unlabeled.to(device), x_unlabeled_bar.to(device)

            # Forward pass for labeled data
            extracted_feat_labeled, labeled_pred, z_labeled = model(x_labeled)
            extracted_feat_labeled_bar, labeled_pred_bar, z_labeled_bar = model(x_labeled_bar)
            
            # Cross-entropy loss for labeled data
            ce_loss = criterion_ce(labeled_pred, label_labeled)

            # Forward pass for unlabeled data
            extracted_feat, labeled_pred_unlabeled, z_unlabeled = model(x_unlabeled)
            extracted_feat_bar, labeled_pred_unlabeled_bar, z_unlabeled_bar = model(x_unlabeled_bar)

            prob = feat2prob(z_unlabeled, model.center)
            prob_bar = feat2prob(z_unlabeled_bar, model.center)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx_unlabeled].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # Contrastive loss using internal projector (combine labeled and unlabeled)
            z_i_labeled = model.projector_CL(extracted_feat_labeled)
            z_j_labeled = model.projector_CL(extracted_feat_labeled_bar)
            z_i_unlabeled = model.projector_CL(extracted_feat)
            z_j_unlabeled = model.projector_CL(extracted_feat_bar)
            
            # Combine features for contrastive learning
            z_i_combined = torch.cat([z_i_labeled, z_i_unlabeled], dim=0)
            z_j_combined = torch.cat([z_j_labeled, z_j_unlabeled], dim=0)
            contrastive_loss = simCLR_loss(z_i_combined, z_j_combined)

            # Pairwise BCE label via ranking (only for unlabeled data)
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            # Total loss combining all components
            loss = ce_loss + sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss
            loss_record.update(loss.item(), x_labeled.size(0) + x_unlabeled.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        
        # Evaluate on unlabeled data
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        # Also evaluate on labeled data for monitoring
        if labeled_eval_loader is not None:
            acc_labeled = test_labeled(model, labeled_eval_loader, args)
            print(f"Labeled Test Accuracy: {acc_labeled:.4f}")

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model
    torch.save({'state_dict': model.state_dict(), 'center': model.center}, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="Accuracy")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')


def test_labeled(model, test_loader, args):
    """Test function for labeled data using cross-entropy accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(device), label.to(device)
            
            _, labeled_pred, _ = model(x)
            
            _, pred = labeled_pred.max(1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Labeled Test Accuracy: {accuracy:.4f}%')
    return accuracy


def test(model, test_loader, args):
    model.eval()

    preds=np.array([])
    targets=np.array([])
    probs= np.zeros((len(test_loader.dataset), args.n_unlabeled_classes))

    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)

        _, _, z_unlabeled = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)

        prob = feat2prob(z_unlabeled, model.center) # get the probability distribution by calculating distance from the center

        _, pred = prob.max(1)

        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()

    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)

    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    probs = torch.from_numpy(probs)

    return acc, nmi, ari, probs 


def plot_tsne(model, test_loader, args):
    """Generates a t-SNE plot of the learned features."""
    model.eval()

    feats = np.zeros((len(test_loader.dataset), 20))
    targets = np.array([])
    
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        
        _, final_feat = model(x)  # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)

        targets = np.append(targets, label.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = final_feat.cpu().detach().numpy()

    # Perform t-SNE on the extracted features
    X_embedded = TSNE(n_components=2).fit_transform(feats)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE Visualization of Learned Features on Unlabeled CIFAR-10 Subset")
    plt.savefig(f"{args.model_folder}/tsne.png")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--warmup_lr', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', default=10, type=int) #10
    parser.add_argument('--epochs', default=100, type=int) #100

    parser.add_argument('--rampup_length', default=5, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=10.0)
    parser.add_argument('--regularization_coeff', type=float, default=.01)

    parser.add_argument('--rampup_length_softBCE', default=10, type=int)
    parser.add_argument('--rampup_coefficient_softBCE', type=float, default=20.0)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--n_unlabeled_classes', default=5, type=int)
    parser.add_argument('--n_labeled_classes', default=5, type=int)

    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--proj_dim_cl', default=128, type=int)
    parser.add_argument('--proj_dim_unlabeled', default=20, type=int)
    parser.add_argument("--imbalance_config", type=str, default=None, help="Class imbalance configuration (e.g., [{'class': 9, 'percentage': 20}, {'class': 7, 'percentage': 5}])")
    
    parser.add_argument('--save_txt', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/cifar10_classif/resnet18_cifar10_classif_5.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--save_txt_name', type=str, default='result.txt')
    parser.add_argument('--DTC', type=str, default='PI')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}/{}'.format(runner_name, args.DTC)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.model_folder = model_dir
    args.model_dir = model_dir+'/'+args.model_name+'.pth'
    args.save_txt_path= args.exp_root+ '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

    print("Arguments: ", args)

    labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.n_labeled_classes))
    labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.n_labeled_classes))

    unlabeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes), imbalance_config=args.imbalance_config)
    unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes))


    model = ResNet(BasicBlock, [2,2,2,2], 5).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir, weights_only=True), strict=False)
    init_feat_extractor = model
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_eval_loader, args)
    args.p_targets = target_distribution(init_probs) 

    model = ResNetMultiHead(BasicBlock, [2, 2, 2, 2], 
                        feat_dim=512, 
                        n_labeled_classes=5, 
                        n_unlabeled_classes=5,
                        proj_dim_cl=128, 
                        proj_dim_unlabeled=20).to(device)
    
    model.encoder.load_state_dict(init_feat_extractor.state_dict(), strict=False)

    model.center.data = torch.tensor(init_centers).float().to(device)
    

    print(model)
    print('---------------------------------')

    for name, param in model.encoder.named_parameters(): 
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    warmup_train(model, unlabeled_train_loader, unlabeled_eval_loader, args)


    if args.DTC == 'PI_CL_softBCE':
        PI_CL_softBCE_train(model, unlabeled_train_loader, unlabeled_eval_loader, args)

    elif args.DTC == 'CE_PI_CL_softBCE':
        CE_PI_CL_softBCE_train(model, 
                               labeled_train_loader, labeled_eval_loader, 
                               unlabeled_train_loader, unlabeled_eval_loader,
                               args)
  

    acc, nmi, ari, _ = test(model, unlabeled_eval_loader, args)

    print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
    print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))

    if args.save_txt:
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))
