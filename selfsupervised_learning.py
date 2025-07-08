from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import  transforms
import pickle
import os
import os.path
import datetime
import numpy as np
from data.simCLRloader import SimCLRDataset
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from utils.util import AverageMeter, accuracy
from models.resnet import BasicBlock
from models.preModel import PreModel
from utils.lars import LARS
from utils.simCLR_loss import SimCLR_Loss
from tqdm import tqdm
import shutil, time, requests, random, copy
from sklearn.manifold import TSNE
from utils.util import cluster_acc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import matplotlib.pyplot as plt
import csv


def get_resnet_blocks(dataset_name):
    """
    Returns the appropriate ResNet block configuration based on dataset
    """
    if dataset_name == 'cifar100':
        return [3, 4, 6, 3]  # ResNet-34
    else:
        return [2, 2, 2, 2]  # ResNet-18 for 

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch, total_epochs):
    model.train()
    tr_loss_epoch = 0

    for step, (data, label) in enumerate(tqdm(train_loader)):
        x_i, x_j = data[0], data[1]
        x_i, x_j = x_i.squeeze().to(device).float(), x_j.squeeze().to(device).float()
        
        optimizer.zero_grad()
        z_i, z_j = model(x_i), model(x_j) # get the model projections
        loss = criterion(z_i, z_j) # simCLR loss
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}] Loss: {round(loss.item(), 5)}")
        
        tr_loss_epoch += loss.item()

    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    avg_tr_loss = tr_loss_epoch / len(train_loader)
    print(f"Epoch [{epoch}/{total_epochs}] Training Loss: {avg_tr_loss:.4f} Learning Rate: {round(lr, 5)}")
    return avg_tr_loss

def test(model, device, valid_loader, criterion, epoch, total_epochs):
    model.eval()
    val_loss_epoch = 0
    
    with torch.no_grad():
        for step, (data, label) in enumerate(tqdm(valid_loader)):
            x_i, x_j = data[0], data[1]
            x_i, x_j = x_i.squeeze().to(device).float(), x_j.squeeze().to(device).float()
            z_i, z_j = model(x_i), model(x_j)
            loss = criterion(z_i, z_j)

            if step % 50 == 0:
                print(f"Step [{step}/{len(valid_loader)}] Loss: {round(loss.item(), 5)}")

            val_loss_epoch += loss.item()
    
    avg_val_loss = val_loss_epoch / len(valid_loader)
    print(f"Epoch [{epoch}/{total_epochs}] Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }, path)

    print(f"Model saved to {path}")

def load_model(model, optimizer, scheduler, path, device):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from {path}, starting from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {path}, starting from scratch")
        start_epoch = 0
    return model, optimizer, scheduler, start_epoch

@torch.no_grad()
def plot_features_And_calculate_metric(model, test_loader, save_path, epoch, device, args):
    torch.manual_seed(1)
    model = model.to(device)
    model.eval()
    targets = np.array([])
    outputs = np.zeros((len(test_loader.dataset), 512 * BasicBlock.expansion)) 
    
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        _, output = model(x)
       
        outputs[idx, :] = output.cpu().detach().numpy()
        targets = np.append(targets, label.cpu().numpy())

    # print("Unique labels:", np.unique(targets))

    pca = PCA(n_components=20) # PCA for dimensionality reduction PCA: 512 -> 20
    pca_features = pca.fit_transform(outputs) # fit the PCA model and transform the features
    kmeans = KMeans(n_clusters=args.n_unlabeled_classes, n_init=20)  # KMeans clustering
    y_pred = kmeans.fit_predict(pca_features)

    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)

    # Normalize targets for categorical mapping
    targets_normalized = (targets - targets.min()).astype(int)  # Map to range 0-19

    # Create t-SNE visualization
    X_embedded = TSNE(n_components=2).fit_transform(outputs)

    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets_normalized, cmap='tab20')
    plt.colorbar(scatter)  # Add color bar to verify mapping
    plt.title(f"t-SNE Visualization of Features on {args.dataset_name} - Epoch {epoch}")
    plt.savefig(f"{save_path}/{args.dataset_name}_epoch{epoch}.png")

    return acc, nmi, ari


def plot_loss(tr_loss, val_loss, save_path):
    plt.figure()
    plt.plot(tr_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path + '/loss_plot.png')
    plt.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--save_interval', type=int, default=50, metavar='I',
                        help='Interval to save the model and plot tsne(default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--model_name', type=str, default='resnet_simCLR')
    parser.add_argument('--load_path', type=str, default='', help='path to load a pretrained model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) 

    dataset_train = SimCLRDataset(
        dataset_name=args.dataset_name,
        split='train',
        dataset_root=args.dataset_root
       )
    dataset_test = SimCLRDataset(
        dataset_name=args.dataset_name,
        split='test',
        dataset_root=args.dataset_root
        )
    

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)
    

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True)
    
    if args.dataset_name == 'cifar10':
        dloader_unlabeled_test = CIFAR10Loader(
            root=args.dataset_root, 
            batch_size=128, 
            split='test', 
            aug=None, 
            shuffle=False, 
            target_list = range(5, 10))
        
        args.n_unlabeled_classes = 5

    elif args.dataset_name == 'cifar100':
        dloader_unlabeled_test = CIFAR100Loader(
            root=args.dataset_root, 
            batch_size=128, 
            split='test', 
            aug=None, 
            shuffle=False, 
            target_list = range(80, 100))
        
        args.n_unlabeled_classes = 20
   
    num_blocks = [2, 2, 2, 2]  # Example for ResNet-18
    model = PreModel(BasicBlock, num_blocks) # Feature Extractor -> Projection Head
    model = model.to(device)

    #OPTMIZER
    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

    # "decay the learning rate with the cosine decay schedule without restarts"
    #SCHEDULER OR LINEAR EWARMUP
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

    #SCHEDULER FOR COSINE DECAY
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

    #LOSS FUNCTION
    criterion = SimCLR_Loss(batch_size = args.batch_size, temperature = 0.5)

    start_epoch = 0
    if args.load_path:
        model, optimizer, mainscheduler, start_epoch = load_model(model, optimizer, mainscheduler, args.load_path, device)

    tr_loss = []
    val_loss = []
    cluser_accs = []

    acc, nmi, ari = plot_features_And_calculate_metric(model.feature_extractor, dloader_unlabeled_test, 
                           model_dir, start_epoch, device, args)
    
    # Append results
    cluser_accs.append([start_epoch, acc, nmi, ari])

    print("-------------------------------------")
    print(f'Epoch-{start_epoch}: ACC = {acc} , NMI = {nmi}, ARI = {ari} ')
    print("-------------------------------------")


    # worst_loss = 101
    

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch}/{args.epochs}]")
        stime = time.time()

        if epoch < 10:
            scheduler = warmupscheduler
        else:
            scheduler = mainscheduler

        # Train
        avg_tr_loss = train(model, device, dloader_train, optimizer, scheduler, criterion, epoch, args.epochs)
        tr_loss.append(avg_tr_loss)

        # Evaluate
        avg_val_loss = test(model, device, dloader_test, criterion, epoch, args.epochs)
        val_loss.append(avg_val_loss)

        # is_best = avg_val_loss < worst_loss 
        # worst_loss = min(avg_val_loss, worst_loss)

        # if is_best and epoch % 10 == 0:
        #     torch.save(model.feature_extractor.state_dict(), args.model_dir)
        #     print(f"Model saved to {args.model_dir}")
    
        print(f"Epoch [{epoch}/{args.epochs}] Training Loss: {avg_tr_loss:.4f}")
        print(f"Epoch [{epoch}/{args.epochs}] Validation Loss: {avg_val_loss:.4f}")

        time_taken = (time.time() - stime) / 60
        print(f"Epoch [{epoch}/{args.epochs}] Time Taken: {time_taken:.2f} minutes")

        # Plot features every 50 epochs
        if epoch > 0 and (epoch+1) % args.save_interval == 0:
            acc, nmi, ari = plot_features_And_calculate_metric(model.feature_extractor, dloader_unlabeled_test, 
                           model_dir, epoch+1, device, args)
            
            cluser_accs.append([epoch + 1, acc, nmi, ari])

            print("-------------------------------------")
            print(f'Epoch-{epoch+1}: ACC = {acc} , NMI = {nmi}, ARI = {ari} ')
            print("-------------------------------------")

            epoch_model_path = os.path.join(model_dir, f'{args.model_name}_epoch{epoch + 1}.pth')
            save_model(model, optimizer, scheduler, epoch, epoch_model_path)
            print(f'Training Checkpoint: {epoch+1} saved')
            
        
     # Plot and save the loss curves
    plot_loss(tr_loss, val_loss, model_dir)

    csv_file = os.path.join(model_dir, "metrics.csv")

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "ACC", "NMI", "ARI"])  # Column headers
        writer.writerows(cluser_accs)  # Write all rows

    print(f"Metrics saved to {csv_file}")

    torch.save(model.feature_extractor.state_dict(), args.model_dir)
    print(f"Model feature extractor saved to {args.model_dir}")



if __name__ == '__main__':
    main()
