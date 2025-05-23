from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import os
import argparse
#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)])  * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()
    

class myBCE(nn.Module):
    
    def forward(self, prob1, prob2, simi):
        # Prepare binary similarity labels
        binary_simi = (simi == 1).float()  # Convert similar (1) to 1, dissimilar (-1) to 0
        mask = simi != 0  # Mask for valid pairs
        
        # Compute pairwise probabilities
        pair_probs = (prob1 * prob2).sum(dim=1)  # Pairwise probabilities

        # Compute traditional BCE loss
        loss = F.binary_cross_entropy(pair_probs[mask], binary_simi[mask], reduction='mean')
        return loss
    
class softBCE_F(nn.Module):

    def forward(self, prob1, prob2, simi):
    
        # Compute pairwise probabilities
        pair_probs = (prob1 * prob2).sum(dim=1)  # Pairwise probabilities

        # Compute traditional BCE loss
        loss = F.binary_cross_entropy(pair_probs, simi, reduction='mean')
        return loss

class softBCE_N(nn.Module):
    def __init__(self):
        super(softBCE_N, self).__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='mean')  # Use BCEWithLogitsLoss

    def forward(self, logits1, logits2, simi):
 
        # Compute pairwise logits (dot product or another similarity measure)
        pair_logits = (logits1 * logits2).sum(dim=1)  # Pairwise logits

        # Compute BCE loss with logits
        loss = self.bce_with_logits(pair_logits, simi)
        return loss
    
class softBCE(nn.Module):
    def __init__(self, use_logits=True):
        """
        Binary cross-entropy for pairwise similarity.

        Args:
            use_logits (bool): 
                - True → expects logits, uses BCEWithLogitsLoss.
                - False → expects probabilities, uses BCELoss.
        """
        super().__init__()
        self.use_logits = use_logits
        if self.use_logits:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, sim1, sim2, pairwise_label):
        """
        Args:
            sim1, sim2: [N, D] tensors of features or probabilities.
            pairwise_label: [N] soft labels between 0 and 1.
        """
        if self.use_logits:
            # Dot product before sigmoid
            similarity = (sim1 * sim2).sum(dim=1)  # shape: [N]
        else:
            # Already in probability form
            similarity = (sim1 * sim2).sum(dim=1).clamp(min=1e-6, max=1 - 1e-6)

        return self.loss_fn(similarity, pairwise_label)

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
