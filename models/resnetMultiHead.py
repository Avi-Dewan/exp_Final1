
import torch
import torch.nn as nn

from .resnet import ResNet
from .preModel import ProjectionHead

class ResNetMultiHead(nn.Module):
    def __init__(self, block, layers, 
                 feat_dim=512, 
                 n_labeled_classes=5, 
                 n_unlabeled_classes=5,
                 proj_dim_cl=128, 
                 proj_dim_unlabeled=20):
        super().__init__()
        
        self.encoder = ResNet(block, layers, n_labeled_classes)  # Keep the linear head!
        
        self.projector_CL = ProjectionHead(feat_dim * block.expansion, 2048, proj_dim_cl)
        self.projector_unlabeled =  nn.Linear(512*block.expansion, proj_dim_unlabeled)
        

    def forward(self, x, return_last =True):

        extracted_feat, final_feat = self.encoder(x)  # extracted_feat = fina final_feat = output of encoder.linear

        labeled_pred = final_feat  # classifier output (num_classes)

        if return_last:
            z_unlabeled = self.projector_unlabeled(extracted_feat)

            return extracted_feat, labeled_pred, z_unlabeled

        return extracted_feat
