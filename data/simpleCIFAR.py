import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_simple_data_loader(dataset='C10', data_root='./datasets', 
                           batch_size=200, num_workers=8, 
                           shuffle=True, pin_memory=True, 
                           drop_last=True):
    
    # Mean and standard deviation for normalization
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    
    # Dataset-specific settings
    if dataset == 'C10':
        which_dataset = datasets.CIFAR10
    elif dataset == 'C100':
        which_dataset = datasets.CIFAR100
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    # Transform to convert images to tensors and normalize
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # Load the dataset
    train_set = which_dataset(root=data_root, train=True, download=True, transform=train_transform)

    # DataLoader setup
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory, 'drop_last': drop_last}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    return train_loader