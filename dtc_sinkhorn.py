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
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool, PairEnum, BCE, myBCE, softBCE_F, softBCE_N
from utils import ramps 
from models.resnet import ResNet, BasicBlock 
from models.preModel import ProjectionHead
from modules.module import feat2prob, target_distribution 
from data.cifarloader import CIFAR10Loader
from utils.simCLR_loss import SimCLR_Loss
from tqdm import tqdm
import numpy as np
import warnings
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
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
        _, extracted_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat), Since, here our linear layer is identity. Extracted features and final features are same
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
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Warmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
    args.p_targets = target_distribution(probs)  # update the target distribution

def Baseline_train(model, train_loader, eva_loader, args):
    '''
    Baseline Training:
    Anneal probability distribution to target distribution.
    At each update interval , update the target distribution
    '''
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs) # update the target distribution
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def PI_train(model, train_loader, eva_loader, args):
    '''
    Sharpening the probability distribution and enforcing consistency with different augmentations
    '''

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    # Lists to store metrics for each epoch
    accuracies = []
    nmi_scores = []
    ari_scores = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            _, final_feat_bar = model(x_bar)  # get the feature of the augmented image
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            prob_bar = feat2prob(final_feat_bar, model.center) #  get the probability distribution of the augmented image
           
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            consistency_loss = F.mse_loss(prob, prob_bar)  # calculate the mean squared error loss between the probability distribution and the probability distribution of the augmented image

            # Why MSE ? 

            loss = sharp_loss + w * consistency_loss   # calculate the total loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        acc, nmi, ari, probs = test(model, eva_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)  # update the target distribution
    # Create a dictionary that includes the model's state dictionary and the center
    model_dict = {'state_dict': model.state_dict(), 'center': model.center}

    # Save the dictionary
    torch.save(model_dict, args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    plt.figure(figsize=(10, 6))
    epochs_range = range(args.epochs)
    plt.plot(epochs_range, accuracies, label="Accuracy")
    plt.plot(epochs_range, nmi_scores, label="NMI")
    plt.plot(epochs_range, ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics over Epochs")
    plt.legend()
    plt.savefig(args.model_folder+'/accuracies.png')

def TE_train(model, train_loader, eva_loader, args):
    '''
     Sharpening the probability distribution and Temporal Ensembling (TE) , calculate the temporal average of the probability distribution and enforce consistency with the temporal average
    '''
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # intermediate values
    z_ema = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)  # current outputs
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features, Reduced Features
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            z_epoch[idx, :] = prob # store the probability distribution
            prob_bar = Variable(z_ema[idx, :], requires_grad=False) # get the temporal average of the probability distribution
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            consistency_loss = F.mse_loss(prob, prob_bar) # calculate the mean squared error loss between the probability distribution and the temporal average of the probability distribution
            loss = sharp_loss + w * consistency_loss  # calculate the total loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = alpha * Z + (1. - alpha) * z_epoch  # calculate the intermediate values
        z_ema = Z * (1. / (1. - alpha ** (epoch + 1)))  # calculate the temporal average of the probability distribution
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)  # update the target distribution
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def TEP_train(model, train_loader, eva_loader, args):
    '''
    Temporal Ensembling with Perturbations (TEP), # Sharpening the probability distribution and does not include a consistency loss and updates the target distribution based on the temporal outputs.
    '''
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # intermediate values
    z_bars = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)  # current outputs
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features, Reduced Features
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        z_epoch = probs.float().to(device) # store the probability distribution
        Z = alpha * Z + (1. - alpha) * z_epoch # calculate the intermediate values
        z_bars = Z * (1. / (1. - alpha ** (epoch + 1)))  # calculate the temporal average of the probability distribution
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(z_bars)  # update the target distribution from the temporal average of the probability distribution instead of the probability distribution
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def PI_TE_train(model, train_loader, eva_loader, args):
    '''
     Sharpening the probability distribution and Temporal Ensembling (TE) , calculate the temporal average of the probability distribution and enforce consistency with the temporal average
    '''
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # intermediate values
    z_ema = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)        # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_unlabeled_classes).float().to(device)  # current outputs
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            _, final_feat = model(x) # model.forward() returns two values: Extracted Features, Reduced Features
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
        
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution

            _, final_feat_bar = model(x_bar)  # get the feature of the augmented image
            prob_bar = feat2prob(final_feat_bar, model.center) #  get the probability distribution of the augmented image
            consistency_loss_PI = F.mse_loss(prob, prob_bar)  # calculate the mean squared error loss between the probability distribution and the probability distribution of the augmented image

            z_epoch[idx, :] = prob # store the probability distribution
            prob_bar_temp = Variable(z_ema[idx, :], requires_grad=False) # get the temporal average of the probability distribution
            consistency_loss_TE = F.mse_loss(prob, prob_bar_temp) # calculate the mean squared error loss between the probability distribution and the temporal average of the probability distribution

            loss = sharp_loss + consistency_loss_PI  +  w * consistency_loss_TE  # calculate the total loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = alpha * Z + (1. - alpha) * z_epoch  # calculate the intermediate values
        z_ema = Z * (1. / (1. - alpha ** (epoch + 1)))  # calculate the temporal average of the probability distribution
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        _, _, _, probs = test(model, eva_loader, args)
        if epoch % args.update_interval ==0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)  # update the target distribution
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


def PI_CL_softBCE_train(model, train_loader, eva_loader, args):
    '''
    Sharpening the probability distribution and enforcing consistency with different augmentations
    '''

    simCLR_loss = SimCLR_Loss(batch_size = args.batch_size, temperature = 0.5).to(device)
    projector = ProjectionHead(512 * BasicBlock.expansion, 2048, 128).to(device)
    criterion_bce = softBCE_N()

    optimizer = SGD(list(model.parameters()) + list(projector.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    w_softBCE = 0
    # Lists to store metrics for each epoch
    accuracies = []
    nmi_scores = []
    ari_scores = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)  # co_eff = 10, length = 5
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)  # co_eff = 20, length = 10
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            extracted_feat, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            extracted_feat_bar, final_feat_bar = model(x_bar)  # get the feature of the augmented image
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            prob_bar = feat2prob(final_feat_bar, model.center) #  get the probability distribution of the augmented image
           
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            consistency_loss = F.mse_loss(prob, prob_bar)  # calculate the mean squared error loss between the probability distribution and the probability distribution of the augmented image

            # simCLR loss
            z_i, z_j = projector(extracted_feat), projector(extracted_feat_bar) 
            contrastive_loss = simCLR_loss(z_i, z_j)


            #BCE
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True) # [68, 512] --> index of features. sorted by value
            rank_idx1, rank_idx2= PairEnum(rank_idx) # [68*68, 512], [68*68, 512]
            rank_idx1, rank_idx2=rank_idx1[:, :args.topk], rank_idx2[:, :args.topk] # [68*68, 5], [68*68, 5]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1) # [68*68, 5]
            rank_idx2, _ = torch.sort(rank_idx2, dim=1) # [68*68, 5]

            # rank_diff = rank_idx1 - rank_idx2 # [68*68, 5]
            # rank_diff = torch.sum(torch.abs(rank_diff), dim=1) # [68*68]
            # pairwise_pseudo_label = torch.ones_like(rank_diff).float().to(device) # [68*68] of one
            # pairwise_pseudo_label[rank_diff>0] = -1  #  [68*68] [if rank_diff is not zero it is -1 (pairwise psuedo label = 0), else it remains 1 ( pairwise psuedo label = 1 ) ]
            
            # Expand rank_idx1 and rank_idx2 for broadcasting
            rank_idx1 = rank_idx1.unsqueeze(2) # Shape: [68*68, 5, 1]
            rank_idx2 = rank_idx2.unsqueeze(1) # Shape: [68*68, 1, 5]
            # Compare all elements and count matches
            matches = (rank_idx1 == rank_idx2).sum(dim=2)  # Shape: [68*68, 5]
            common_elements = matches.sum(dim=1)          # Shape: [68*68]
            # Calculate soft pseudo-label
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)


            loss = sharp_loss + w * consistency_loss + w*contrastive_loss +  w_softBCE*bce_loss # calculate the total loss
            # loss = sharp_loss + w * consistency_loss  + bce_loss # calculate the total loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        acc, nmi, ari, probs = test(model, eva_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)  # update the target distribution
    # Create a dictionary that includes the model's state dictionary and the center
    model_dict = {'state_dict': model.state_dict(), 'center': model.center}

    # Save the dictionary
    torch.save(model_dict, args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    plt.figure(figsize=(10, 6))
    epochs_range = range(args.epochs)
    plt.plot(epochs_range, accuracies, label="Accuracy")
    plt.plot(epochs_range, nmi_scores, label="NMI")
    plt.plot(epochs_range, ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics over Epochs")
    plt.legend()
    plt.savefig(args.model_folder+'/accuracies.png')   


def PI_CL_softBCE_sinkhorn_train(model, train_loader, eva_loader, args):
    '''
    Sharpening the probability distribution and enforcing consistency with different augmentations
    '''

    simCLR_loss = SimCLR_Loss(batch_size = args.batch_size, temperature = 0.5).to(device)
    projector = ProjectionHead(512 * BasicBlock.expansion, 2048, 128).to(device)
    criterion_bce = softBCE_N()

    optimizer = SGD(list(model.parameters()) + list(projector.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    w = 0
    w_softBCE = 0
    # Lists to store metrics for each epoch
    accuracies = []
    nmi_scores = []
    ari_scores = []

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)  # co_eff = 10, length = 5
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)  # co_eff = 20, length = 10
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            extracted_feat, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
            extracted_feat_bar, final_feat_bar = model(x_bar)  # get the feature of the augmented image
            prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
            prob_bar = feat2prob(final_feat_bar, model.center) #  get the probability distribution of the augmented image
           
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device)) # calculate the KL divergence loss between the probability distribution and the target distribution
            consistency_loss = F.mse_loss(prob, prob_bar)  # calculate the mean squared error loss between the probability distribution and the probability distribution of the augmented image

            # simCLR loss
            z_i, z_j = projector(extracted_feat), projector(extracted_feat_bar) 
            contrastive_loss = simCLR_loss(z_i, z_j)


            #BCE
            B, C = prob.shape
            #  stack them into a single [2B × C] matrix
            all_probs = torch.cat([prob, prob_bar], dim=0)      # [2B, C]

            #  apply Sinkhorn to get balanced soft assignments
            sink = SinkhornKnopp()
            # sink = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk)
            all_pseudo = sink(all_probs)                       # [2B, C], each row sums to 1

            # 1d) split back into two views
            pseudo, pseudo_bar = all_pseudo[:B], all_pseudo[B:] # each [B, C]
                        # Enumerate all pairs (i,j) for view1 and view2
            pseudo_i, pseudo_j = PairEnum(pseudo)            # both [B*B, C]
            pseudo_bar_i, pseudo_bar_j = PairEnum(pseudo_bar)

            # Soft pairwise label = dot-product of the two soft cluster distributions
            # (i.e. probability that i and j belong to the same cluster)
            pairwise_pseudo_label = (pseudo_i * pseudo_j).sum(dim=1)           # [B*B]
            pairwise_pseudo_label_bar = (pseudo_bar_i * pseudo_bar_j).sum(dim=1)
            # you could average these two, or use just one view’s labels:
            pairwise_pseudo_label = 0.5 * (pairwise_pseudo_label + pairwise_pseudo_label_bar)
            
            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)


            loss = sharp_loss + w * consistency_loss + w*contrastive_loss +  w_softBCE*bce_loss # calculate the total loss
            # loss = sharp_loss + w * consistency_loss  + bce_loss # calculate the total loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        acc, nmi, ari, probs = test(model, eva_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)  # update the target distribution
    # Create a dictionary that includes the model's state dictionary and the center
    model_dict = {'state_dict': model.state_dict(), 'center': model.center}

    # Save the dictionary
    torch.save(model_dict, args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    plt.figure(figsize=(10, 6))
    epochs_range = range(args.epochs)
    plt.plot(epochs_range, accuracies, label="Accuracy")
    plt.plot(epochs_range, nmi_scores, label="NMI")
    plt.plot(epochs_range, ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("Training Metrics over Epochs")
    plt.legend()
    plt.savefig(args.model_folder+'/accuracies.png')   

def test(model, test_loader, args):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    probs= np.zeros((len(test_loader.dataset), args.n_unlabeled_classes))
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        _, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
        prob = feat2prob(final_feat, model.center) # get the probability distribution by calculating distance from the center
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

def plot_pdf(model, test_loader, args):
    """Generates PDF plots for intermediate features, final outputs, and a combined overlay plot."""
    model.eval()
    extracted_feats = np.zeros((len(test_loader.dataset), 512))
    final_feats = np.zeros((len(test_loader.dataset), 20))
    
    for batch_idx, (x, _, idx) in enumerate(tqdm(test_loader)):
        x = x.to(device)
        
        extracted_feat, final_feat = model(x) # model.forward() returns two values: Extracted Features(extracted_feat), Final Features(final_feat)
        
        idx = idx.data.cpu().numpy()
        
        extracted_feats[idx, :] = extracted_feat.cpu().detach().numpy()
        final_feats[idx, :] = final_feat.cpu().detach().numpy()

    # Plot individual PDFs for intermediate features and final outputs
    plt.figure(figsize=(12, 6))

    # PDF for intermediate features
    plt.subplot(1, 2, 1)
    sns.kdeplot(extracted_feats.flatten(), bw_adjust=0.5, color='blue')
    plt.title("PDF of Intermediate Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    
    # PDF for final outputs
    plt.subplot(1, 2, 2)
    sns.kdeplot(final_feats.flatten(), bw_adjust=0.5, color='green')
    plt.title("PDF of Final Outputs")
    plt.xlabel("Output Value")
    plt.ylabel("Density")
    
    plt.tight_layout()
    plt.savefig(f"{args.model_folder}/pdf_individual.png")
    plt.show()
    
    # Combined PDF overlay
    plt.figure(figsize=(8, 6))
    sns.kdeplot(extracted_feats.flatten(), bw_adjust=0.5, color='blue', label="Intermediate Features")
    sns.kdeplot(final_feats.flatten(), bw_adjust=0.5, color='green', label="Final Outputs")
    plt.title("Combined PDF of Intermediate Features and Final Outputs")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{args.model_folder}/pdf_combined.png")
    plt.show()
    print('pdf plots saved in ${args.model_folder}')

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
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--seed', default=1, type=int)
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

    train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes))
    eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range(args.n_labeled_classes, args.n_labeled_classes+args.n_unlabeled_classes))


    model = ResNet(BasicBlock, [2,2,2,2], 5).to(device)
    model.load_state_dict(torch.load(args.pretrain_dir, weights_only=True), strict=False)
    # model.load_state_dict(torch.load(args.pretrain_dir), strict=False)
    model.linear= Identity()
    init_feat_extractor = model
    init_acc, init_nmi, init_ari, init_centers, init_probs = init_prob_kmeans(init_feat_extractor, eval_loader, args)
    args.p_targets = target_distribution(init_probs) 



    # model = ResNet(BasicBlock, [2,2,2,2], args.n_unlabeled_classes).to(device)
    model = ResNet(BasicBlock, [2,2,2,2], 20).to(device)
    model.load_state_dict(init_feat_extractor.state_dict(), strict=False)
    # model.center= Parameter(torch.Tensor(args.n_unlabeled_classes, args.n_unlabeled_classes))
    model.center= Parameter(torch.Tensor(args.n_unlabeled_classes, 20))
    model.center.data = torch.tensor(init_centers).float().to(device)

    print(model)
    print('---------------------------------')
    for name, param in model.named_parameters(): 
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    warmup_train(model, train_loader, eval_loader, args)


    if args.DTC == 'Baseline':
        Baseline_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'PI':
        PI_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'TE':
        TE_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'TEP':
        TEP_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'PI_TE':
        PI_TE_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'PI_CL_softBCE':
        PI_CL_softBCE_train(model, train_loader, eval_loader, args)
    elif args.DTC == 'sinkhorn_softBCE':
        PI_CL_softBCE_sinkhorn_train(model, train_loader, eval_loader, args)
  
    # Final ACC and plot tsne and pdf
    acc, nmi, ari, _ = test(model, eval_loader, args)
    # plot_tsne(model, eval_loader, args)
    # plot_pdf(model, eval_loader, args)

    print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
    print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))
    if args.save_txt:
        with open(args.save_txt_path, 'a') as f:
            f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))

    # print("Testing if properly saved")
    # # Load the dictionary
    # model_dict = torch.load(args.model_dir)

    # # Create the model with clusters
    # model = ResNet(BasicBlock, [2,2,2,2], args.n_unlabeled_classes).to(device)

    # # Load the state dictionary into the model
    # model.load_state_dict(model_dict['state_dict'], strict=False)

    # # Load the center
    # model.center = Parameter(model_dict['center'])

    # acc, nmi, ari, _ = test(model, eval_loader, args,False)
    # print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))