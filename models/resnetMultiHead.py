
import torch
import torch.nn as nn

from .resnet import ResNet
from .preModel import ProjectionHead

class ResNetMultiHead(nn.Module):
    def __init__(self, block, layers, feat_dim=512, num_classes=5, 
                 proj_dim_cl=128, proj_dim_unlabeled=20):
        super().__init__()
        
        self.encoder = ResNet(block, layers, num_classes)  # Keep the linear head!
        
        self.projector_CL = ProjectionHead(feat_dim * block.expansion, 2048, proj_dim_cl)
        self.projector_unlabeled =  nn.Linear(512*block.expansion, num_classes)

        # Learnable cluster centers for Sinkhorn etc.
        self.center = nn.Parameter(torch.randn(proj_dim_unlabeled, proj_dim_unlabeled))

    def forward(self, x, return_all=False):

        extracted_feat, final_feat = self.encoder(x)  # extracted_feat = fina final_feat = output of encoder.linear

        labeled_pred = final_feat  # classifier output (num_classes)

        if return_all:
            z_cl = self.projector_CL(extracted_feat)
            z_unlabeled = self.projector_unlabeled(extracted_feat)

            return extracted_feat, labeled_pred, z_unlabeled, z_cl

        return extracted_feat, labeled_pred
