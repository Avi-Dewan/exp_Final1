import torch
import csv
import datetime
import json
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.enhanced_sinkhorn import ClassAwareSinkhornKnopp
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, softBCE, str2bool, PairEnum, BCE, myBCE, softBCE_F, softBCE_N
from utils import ramps 
from utils.simCLR_loss import SimCLR_Loss
from utils.sinkhorn_knopp import SinkhornKnopp

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

@torch.no_grad()
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

    for batch_idx, (x, label, idx) in enumerate(tqdm(eval_loader)):
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

def init_labeled_clusters(model, labeled_loader, args):
    """
    Initialize cluster centers using labeled data.
    """
    torch.manual_seed(1)
    model = model.to(device)
    
    model.eval()

    targets = np.zeros(len(labeled_loader.dataset))  # Store labels
    extracted_features = np.zeros((len(labeled_loader.dataset), 512))  # Store features

    # Extract features for the labeled data
    for _, (x, label, idx) in enumerate(labeled_loader):
        x = x.to(device)
        extracted_feat, _ = model(x)  # Extract features (using the model)
    
        idx = idx.data.cpu().numpy()  # Get indices
        extracted_features[idx, :] = extracted_feat.data.cpu().numpy()  # Store the features
        targets[idx] = label.data.cpu().numpy()  # Store the labels

     # pca = PCA(n_components=args.n_unlabeled_classes)
    pca = PCA(n_components=20) # PCA for dimensionality reduction PCA: 512 -> 20

    extracted_features = pca.fit_transform(extracted_features) # fit the PCA model and transform the features

    # Now, calculate the center for each labeled class
    labeled_centers = np.zeros((args.n_labeled_classes, extracted_features.shape[1]))

    for i in range(args.n_labeled_classes):
        labeled_centers[i] = np.mean(extracted_features[targets == i], axis=0)

    # Convert to tensor and move to GPU if needed
    labeled_centers = torch.tensor(labeled_centers).to(device)

    return labeled_centers

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
            prob = feat2prob(z_unlabeled, model.encoder.center) # get the probability distribution by calculating distance from the center
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        
    args.p_targets = target_distribution(probs)  # update the target distribution


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
    ce_criterion = nn.CrossEntropyLoss()
    sinkhorn = SinkhornKnopp(num_iters=3, epsilon=0.05)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores, f1_scores = [], [], [], []
    
    # Initialize best metrics tracking
    best_acc = 0.0
    best_nmi = 0.0
    best_ari = 0.0
    best_f1 = 0.0
    best_epoch_acc = 0
    best_epoch_nmi = 0
    best_epoch_ari = 0
    best_epoch_f1 = 0
    
    for epoch in range(args.epochs):
        model.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)

        labeled_loader_iter = iter(labeled_train_loader)
        
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, _ , z_unlabeled = model(x)
            extracted_feat_bar, _ , z_unlabeled_bar = model(x_bar)

            prob = feat2prob(z_unlabeled, model.encoder.center)
            prob_bar = feat2prob(z_unlabeled_bar, model.encoder.center)

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

             # Step 1: Raw logits
            temp = 0.1
            logits = -torch.sum((z_unlabeled.unsqueeze(1) - model.encoder.center) ** 2, dim=2) 
            logits_bar = -torch.sum((z_unlabeled_bar.unsqueeze(1) - model.encoder.center) ** 2, dim=2) 
            

            # Step 2: Sinkhorn input stability
            logits_all = torch.cat([logits, logits_bar], dim=0)
            logits_all = logits_all - logits_all.max()  # stability
         
            # Step 3: Sinkhorn assignments
            pseudo_all = sinkhorn(logits_all)
            pseudo, pseudo_bar = pseudo_all[:logits.size(0)], pseudo_all[logits.size(0):]

            # Step 4: Pairwise soft labels
            pseudo_i, pseudo_j = PairEnum(pseudo)
            pseudo_bar_i, pseudo_bar_j = PairEnum(pseudo_bar)

            # === Sinkhorn pseudo-label (s_label)
            s_label = 0.5 * (
                (pseudo_i * pseudo_j).sum(dim=1) +
                (pseudo_bar_i * pseudo_bar_j).sum(dim=1)
            )

            # === Ranking-based pseudo-label (r_label)
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            r_label = common_elements.float() / args.topk

            # === Final combined pseudo-label
            # alpha = args.mix_alpha  # new argument, e.g. 0.5
            alpha = 0.5  # new argument, e.g. 0.5

            pairwise_pseudo_label = (1 - alpha) * r_label + alpha * s_label
            pairwise_pseudo_label = pairwise_pseudo_label.clamp(min=1e-4, max=1 - 1e-4)

          
            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)


            # Step 6: Final BCE
            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            # === Add labeled CE loss ===
            try:
                x_l, y_l, _ = next(labeled_loader_iter)

            except StopIteration:
                labeled_loader_iter = iter(labeled_train_loader)
                x_l, y_l, _ = next(labeled_loader_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            _, labeled_pred_l, _ = model(x_l)

            ce_loss = ce_criterion(labeled_pred_l, y_l)

            
            loss =ce_loss + sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        f1 = test_labeled(model, labeled_eval_loader)
        
        # Track best metrics
        if acc > best_acc:
            best_acc = acc
            best_epoch_acc = epoch
            
        if nmi > best_nmi:
            best_nmi = nmi
            best_epoch_nmi = epoch
            
        if ari > best_ari:
            best_ari = ari
            best_epoch_ari = epoch

        if f1 > best_f1:
            best_f1 = f1
            best_epoch_f1 = epoch

        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        f1_scores.append(f1)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

     # Print best metrics summary
    print("\n" + "="*60)
    print("BEST METRICS SUMMARY:")
    print("="*60)
    print(f"Best ACC: {best_acc:.4f} (achieved at epoch {best_epoch_acc})")
    print(f"Best NMI: {best_nmi:.4f} (achieved at epoch {best_epoch_nmi})")
    print(f"Best ARI: {best_ari:.4f} (achieved at epoch {best_epoch_ari})")
    print(f"Best f1: {best_f1:.4f} (achieved at epoch {best_epoch_f1})")
    print("="*60)


    # Save model
    torch.save({'state_dict': model.state_dict(), 'center': model.encoder.center}, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="ACC")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.plot(range(args.epochs), f1_scores, label="F1(labeled data)")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')

    return best_acc, best_nmi, best_ari, best_f1

@torch.no_grad()
def test(model, test_loader, args):
    model.eval()

    preds=np.array([])
    targets=np.array([])
    probs= np.zeros((len(test_loader.dataset), args.n_unlabeled_classes))

    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)

        _, _, z_unlabeled = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)

        prob = feat2prob(z_unlabeled, model.encoder.center) # get the probability distribution by calculating distance from the center

        _, pred = prob.max(1)

        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()

    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)

    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    probs = torch.from_numpy(probs)

    return acc, nmi, ari, probs 

@torch.no_grad()
def test_labeled(model, test_loader):
    model.eval()
    preds = []
    targets = []
    
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)

        _, output, _ = model(x)  # model forward pass (gives two output: extracted features and final output)
        _, pred = output.max(1) 
        pred = pred

        targets.extend(label.cpu().numpy())
        preds.extend(pred.cpu().numpy())
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='macro')
    recall = recall_score(targets, preds, average='macro')
    f1 = f1_score(targets, preds, average='macro')

    # Print metrics in a single line
    print('Labeld Test Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(accuracy, precision, recall, f1))

    return f1

@torch.no_grad()
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


def log_results(log_file, config_name, imbalance_config, final_acc, final_nmi, final_ari, best_acc, best_nmi, best_ari, best_f1):
    """
    Log training results to CSV file
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(log_file)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert imbalance_config to string if it's a list/dict
    if isinstance(imbalance_config, (list, dict)):
        imbalance_config_str = json.dumps(imbalance_config).replace('"', "'")
    else:
        imbalance_config_str = str(imbalance_config)
    
    # Write to log file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'timestamp', 'config_name', 'imbalance_config', 
                'final_acc', 'final_nmi', 'final_ari', 
                'best_acc', 'best_nmi', 'best_ari', 'best_f1'
            ])
        
        # Write data row
        writer.writerow([
            timestamp, config_name, imbalance_config_str,
            f"{final_acc:.4f}", f"{final_nmi:.4f}", f"{final_ari:.4f}",
            f"{best_acc:.4f}", f"{best_nmi:.4f}", f"{best_ari:.4f}", f"{best_f1:.4f}"
        ])
    
    print(f"Results logged to: {log_file}")


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
    
     # Add new arguments for logging
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file for results')
    parser.add_argument('--config_name', type=str, default='default', help='Name of the configuration being run')

    parser.add_argument('--save_txt', default=False, type=str2bool, help='save txt or not', metavar='BOOL')
    parser.add_argument('--pretrain_dir', type=str, default='./data/experiments/cifar10_classif/resnet18_cifar10_classif_5.pth')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--save_txt_name', type=str, default='result.txt')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}'.format(runner_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.model_folder = model_dir
    args.model_dir = model_dir+'/'+args.model_name+'.pth'
    args.save_txt_path= args.exp_root+ '{}/{}'.format(runner_name, args.save_txt_name)

    print("Arguments: ", args)

    labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.n_labeled_classes))
    labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.n_labeled_classes))

    unlabeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes), imbalance_config=args.imbalance_config)
    unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes), imbalance_config=args.imbalance_config)


    model = ResNet(BasicBlock, [2,2,2,2], 5).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir, weights_only=True), strict=False)
    init_feat_extractor = model
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, unlabeled_eval_loader, args)
    args.p_targets = target_distribution(init_probs) 

    init_labeled_centers = init_labeled_clusters(init_feat_extractor, labeled_train_loader, args)

    model = ResNetMultiHead(BasicBlock, [2, 2, 2, 2], 
                        feat_dim=512, 
                        n_labeled_classes=5, 
                        n_unlabeled_classes=5,
                        proj_dim_cl=128, 
                        proj_dim_unlabeled=20).to(device)
    
    model.encoder.load_state_dict(init_feat_extractor.state_dict(), strict=False)

    model.encoder.center = Parameter(torch.Tensor(args.n_unlabeled_classes, 20))
    model.encoder.center.data = torch.tensor(init_centers).float().to(device)


    model.encoder.labeledCenter = Parameter(torch.Tensor(args.n_labeled_classes, 20))
    model.encoder.labeledCenter.data = torch.tensor(init_labeled_centers).float().to(device)
    
    print('---------------------------------')
    print(model)
    print('---------------------------------')

    for name, param in model.encoder.named_parameters(): 
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    warmup_train(model, unlabeled_train_loader, unlabeled_eval_loader, args)

    # Main training and get best metrics
    best_acc, best_nmi, best_ari, best_f1 = CE_PI_CL_softBCE_train(model, 
                            labeled_train_loader, labeled_eval_loader, 
                            unlabeled_train_loader, unlabeled_eval_loader,
                            args)
    
    acc, nmi, ari, _ = test(model, unlabeled_eval_loader, args)

    print('\n' + '='*60)
    print('FINAL RESULTS SUMMARY:')
    print('='*60)
    print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
    print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))
    print('Best ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(best_acc, best_nmi, best_ari))
    print('Best labeled f1 sore {:.4f}'.format(best_f1))
    print('='*60)

    # Log results if log_file is provided
    if args.log_file:
        log_results(
            args.log_file, 
            args.config_name, 
            args.imbalance_config, 
            acc, nmi, ari, 
            best_acc, best_nmi, best_ari, best_f1
        )

    if args.save_txt:
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))(x, _)