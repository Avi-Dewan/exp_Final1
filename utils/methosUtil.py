import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules.module import feat2prob, target_distribution 

class SeparatedClusterHead(nn.Module):
    """
    Maintains separate cluster centers for labeled and unlabeled data
    """
    def __init__(self, feat_dim, n_labeled_classes, n_unlabeled_classes):
        super().__init__()
        self.n_labeled = n_labeled_classes
        self.n_unlabeled = n_unlabeled_classes
        
        # Separate cluster centers
        self.labeled_centers = Parameter(torch.Tensor(n_labeled_classes, feat_dim))
        self.unlabeled_centers = Parameter(torch.Tensor(n_unlabeled_classes, feat_dim))
        
        # Initialize centers
        nn.init.xavier_uniform_(self.labeled_centers)
        nn.init.xavier_uniform_(self.unlabeled_centers)
    
    def forward(self, features, is_labeled=False):
        if is_labeled:
            return feat2prob(features, self.labeled_centers)
        else:
            return feat2prob(features, self.unlabeled_centers)

def separation_loss(labeled_centers, unlabeled_centers, margin=2.0):
    """
    Encourage separation between labeled and unlabeled cluster centers
    """
    # Compute pairwise distances between all labeled and unlabeled centers
    labeled_expanded = labeled_centers.unsqueeze(1)  # [n_labeled, 1, feat_dim]
    unlabeled_expanded = unlabeled_centers.unsqueeze(0)  # [1, n_unlabeled, feat_dim]
    distances = torch.norm(labeled_expanded - unlabeled_expanded, dim=2)  # [n_labeled, n_unlabeled]
    
    # Margin loss: encourage distances to be at least 'margin'
    separation_loss = F.relu(margin - distances).mean()
    return separation_loss