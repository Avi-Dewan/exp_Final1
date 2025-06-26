"""
Advanced Deep Clustering Methods with Labeled/Unlabeled Cluster Separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

# =============================================================================
# METHOD 1: EXPLICIT CLUSTER CENTER SEPARATION
# =============================================================================

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

def METHOD1_train(model, labeled_loader, unlabeled_loader, args):
    """
    METHOD 1: Explicit Cluster Center Separation
    - Use ground truth labels to create labeled cluster centers
    - Separate unlabeled centers from labeled ones
    - Regular clustering loss for unlabeled data
    """
    
    # Replace the clustering head with separated version
    model.cluster_head = SeparatedClusterHead(
        feat_dim=20, 
        n_labeled_classes=args.n_labeled_classes,
        n_unlabeled_classes=args.n_unlabeled_classes
    ).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.epochs):
        model.train()
        
        labeled_iter = iter(labeled_loader)
        
        for batch_idx, ((x_u, x_u_bar), _, idx_u) in enumerate(unlabeled_loader):
            # Get labeled batch
            try:
                (x_l, _), y_l, idx_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_l, _), y_l, idx_l = next(labeled_iter)
            
            x_u, x_u_bar = x_u.to(device), x_u_bar.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
            # Forward pass
            feat_u, _, z_u = model(x_u)
            feat_u_bar, _, z_u_bar = model(x_u_bar)
            feat_l, pred_l, z_l = model(x_l)
            
            # Labeled cluster assignment (use ground truth)
            prob_l = model.cluster_head(z_l, is_labeled=True)
            labeled_cluster_loss = F.cross_entropy(prob_l, y_l)
            
            # Unlabeled clustering
            prob_u = model.cluster_head(z_u, is_labeled=False)
            prob_u_bar = model.cluster_head(z_u_bar, is_labeled=False)
            
            # Standard clustering losses for unlabeled
            sharp_loss = F.kl_div(prob_u.log(), args.p_targets[idx_u].float().to(device))
            consistency_loss = F.mse_loss(prob_u, prob_u_bar)
            
            # Separation loss between labeled and unlabeled centers
            sep_loss = separation_loss(
                model.cluster_head.labeled_centers,
                model.cluster_head.unlabeled_centers,
                margin=2.0
            )
            
            # Total loss
            loss = labeled_cluster_loss + sharp_loss + consistency_loss + 0.5 * sep_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# =============================================================================
# METHOD 2: REPULSION LOSS - PUSH UNLABELED AWAY FROM LABELED CENTERS
# =============================================================================

def repulsion_loss(unlabeled_features, labeled_centers, temperature=0.1):
    """
    Push unlabeled samples away from labeled cluster centers
    """
    # Compute distances from unlabeled features to labeled centers
    distances = torch.cdist(unlabeled_features, labeled_centers)  # [batch_size, n_labeled_classes]
    
    # Convert to similarities (closer = higher similarity)
    similarities = torch.exp(-distances / temperature)
    
    # We want to minimize these similarities (push away)
    repulsion = similarities.mean()
    
    return repulsion

def METHOD2_train(model, labeled_loader, unlabeled_loader, args):
    """
    METHOD 2: Repulsion-based Training
    - Use labeled data to establish labeled cluster centers
    - Push unlabeled samples away from labeled centers
    - Regular clustering for unlabeled data
    """
    
    # Maintain labeled cluster centers
    labeled_centers = Parameter(torch.Tensor(args.n_labeled_classes, 20)).to(device)
    nn.init.xavier_uniform_(labeled_centers)
    
    optimizer = torch.optim.SGD(
        list(model.parameters()) + [labeled_centers], 
        lr=args.lr, momentum=args.momentum
    )
    
    for epoch in range(args.epochs):
        model.train()
        
        labeled_iter = iter(labeled_loader)
        
        for batch_idx, ((x_u, x_u_bar), _, idx_u) in enumerate(unlabeled_loader):
            # Get labeled batch
            try:
                (x_l, _), y_l, idx_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_l, _), y_l, idx_l = next(labeled_iter)
            
            x_u, x_u_bar = x_u.to(device), x_u_bar.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
            # Forward pass
            feat_u, _, z_u = model(x_u)
            feat_u_bar, _, z_u_bar = model(x_u_bar)
            feat_l, pred_l, z_l = model(x_l)
            
            # Labeled center assignment
            labeled_probs = feat2prob(z_l, labeled_centers)
            labeled_loss = F.cross_entropy(labeled_probs, y_l)
            
            # Unlabeled clustering (standard)
            prob_u = feat2prob(z_u, model.encoder.center)
            prob_u_bar = feat2prob(z_u_bar, model.encoder.center)
            
            sharp_loss = F.kl_div(prob_u.log(), args.p_targets[idx_u].float().to(device))
            consistency_loss = F.mse_loss(prob_u, prob_u_bar)
            
            # Repulsion: push unlabeled samples away from labeled centers
            repulsion = repulsion_loss(z_u, labeled_centers, temperature=0.1)
            
            # Total loss
            loss = labeled_loss + sharp_loss + consistency_loss + 0.3 * repulsion
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# =============================================================================
# METHOD 3: CONTRASTIVE SEPARATION
# =============================================================================

def contrastive_separation_loss(labeled_features, unlabeled_features, temperature=0.5):
    """
    Contrastive loss to separate labeled and unlabeled feature spaces
    """
    # Normalize features
    labeled_norm = F.normalize(labeled_features, dim=1)
    unlabeled_norm = F.normalize(unlabeled_features, dim=1)
    
    # Compute similarities
    sim_matrix = torch.mm(labeled_norm, unlabeled_norm.t()) / temperature
    
    # We want to minimize these similarities (push apart)
    # Use negative log likelihood of pushing apart
    separation_loss = torch.logsumexp(sim_matrix, dim=1).mean()
    
    return separation_loss

def METHOD3_train(model, labeled_loader, unlabeled_loader, args):
    """
    METHOD 3: Contrastive Feature Space Separation
    - Learn separate feature spaces for labeled and unlabeled data
    - Use contrastive loss to push feature spaces apart
    - Dual-head architecture for separate feature learning
    """
    
    # Add separate projection heads
    model.labeled_projector = nn.Linear(512, 128).to(device)
    model.unlabeled_projector = nn.Linear(512, 128).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    for epoch in range(args.epochs):
        model.train()
        
        labeled_iter = iter(labeled_loader)
        
        for batch_idx, ((x_u, x_u_bar), _, idx_u) in enumerate(unlabeled_loader):
            try:
                (x_l, _), y_l, idx_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_l, _), y_l, idx_l = next(labeled_iter)
            
            x_u, x_u_bar = x_u.to(device), x_u_bar.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
            # Forward pass
            feat_u, _, z_u = model(x_u)
            feat_u_bar, _, z_u_bar = model(x_u_bar)
            feat_l, pred_l, z_l = model(x_l)
            
            # Project to separate spaces
            labeled_proj = model.labeled_projector(feat_l)
            unlabeled_proj = model.unlabeled_projector(feat_u)
            
            # Labeled classification
            labeled_loss = F.cross_entropy(pred_l, y_l)
            
            # Unlabeled clustering
            prob_u = feat2prob(z_u, model.encoder.center)
            prob_u_bar = feat2prob(z_u_bar, model.encoder.center)
            
            sharp_loss = F.kl_div(prob_u.log(), args.p_targets[idx_u].float().to(device))
            consistency_loss = F.mse_loss(prob_u, prob_u_bar)
            
            # Contrastive separation
            separation = contrastive_separation_loss(labeled_proj, unlabeled_proj)
            
            # Total loss
            loss = labeled_loss + sharp_loss + consistency_loss + 0.2 * separation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# =============================================================================
# METHOD 4: ADVERSARIAL DOMAIN SEPARATION
# =============================================================================

class DomainDiscriminator(nn.Module):
    """
    Discriminator to distinguish between labeled and unlabeled features
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary: labeled vs unlabeled
        )
    
    def forward(self, x):
        return self.classifier(x)

def METHOD4_train(model, labeled_loader, unlabeled_loader, args):
    """
    METHOD 4: Adversarial Domain Separation
    - Train a discriminator to distinguish labeled vs unlabeled features
    - Train feature extractor to fool the discriminator
    - This encourages separate feature distributions
    """
    
    discriminator = DomainDiscriminator(feat_dim=512).to(device)
    
    # Separate optimizers
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        
        labeled_iter = iter(labeled_loader)
        
        for batch_idx, ((x_u, x_u_bar), _, idx_u) in enumerate(unlabeled_loader):
            try:
                (x_l, _), y_l, idx_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_l, _), y_l, idx_l = next(labeled_iter)
            
            x_u, x_u_bar = x_u.to(device), x_u_bar.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
            # Forward pass
            feat_u, _, z_u = model(x_u)
            feat_u_bar, _, z_u_bar = model(x_u_bar)
            feat_l, pred_l, z_l = model(x_l)
            
            # === Train Discriminator ===
            disc_optimizer.zero_grad()
            
            # Discriminator predictions
            labeled_domain_pred = discriminator(feat_l.detach())
            unlabeled_domain_pred = discriminator(feat_u.detach())
            
            # Domain labels (0 = labeled, 1 = unlabeled)
            labeled_domain_target = torch.zeros(feat_l.size(0), dtype=torch.long).to(device)
            unlabeled_domain_target = torch.ones(feat_u.size(0), dtype=torch.long).to(device)
            
            disc_loss = F.cross_entropy(labeled_domain_pred, labeled_domain_target) + \
                       F.cross_entropy(unlabeled_domain_pred, unlabeled_domain_target)
            
            disc_loss.backward()
            disc_optimizer.step()
            
            # === Train Model (with adversarial loss) ===
            model_optimizer.zero_grad()
            
            # Standard losses
            labeled_loss = F.cross_entropy(pred_l, y_l)
            
            prob_u = feat2prob(z_u, model.encoder.center)
            prob_u_bar = feat2prob(z_u_bar, model.encoder.center)
            
            sharp_loss = F.kl_div(prob_u.log(), args.p_targets[idx_u].float().to(device))
            consistency_loss = F.mse_loss(prob_u, prob_u_bar)
            
            # Adversarial loss: fool the discriminator
            labeled_domain_pred = discriminator(feat_l)
            unlabeled_domain_pred = discriminator(feat_u)
            
            # Flip the labels to fool discriminator
            adversarial_loss = F.cross_entropy(labeled_domain_pred, unlabeled_domain_target[:feat_l.size(0)]) + \
                              F.cross_entropy(unlabeled_domain_pred, labeled_domain_target[:feat_u.size(0)])
            
            # Total model loss
            model_loss = labeled_loss + sharp_loss + consistency_loss + 0.1 * adversarial_loss
            
            model_loss.backward()
            model_optimizer.step()

# =============================================================================
# METHOD 5: HYBRID APPROACH - BEST OF ALL WORLDS
# =============================================================================

def METHOD5_HYBRID_train(model, labeled_loader, unlabeled_loader, args):
    """
    METHOD 5: Hybrid Approach
    Combines multiple strategies:
    - Explicit separation of cluster centers
    - Repulsion loss
    - Contrastive separation
    """
    
    # Separate cluster centers for labeled data
    labeled_centers = Parameter(torch.Tensor(args.n_labeled_classes, 20)).to(device)
    nn.init.xavier_uniform_(labeled_centers)
    
    optimizer = torch.optim.SGD(
        list(model.parameters()) + [labeled_centers], 
        lr=args.lr, momentum=args.momentum
    )
    
    for epoch in range(args.epochs):
        model.train()
        
        labeled_iter = iter(labeled_loader)
        
        for batch_idx, ((x_u, x_u_bar), _, idx_u) in enumerate(unlabeled_loader):
            try:
                (x_l, _), y_l, idx_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_l, _), y_l, idx_l = next(labeled_iter)
            
            x_u, x_u_bar = x_u.to(device), x_u_bar.to(device)
            x_l, y_l = x_l.to(device), y_l.to(device)
            
            # Forward pass
            feat_u, _, z_u = model(x_u)
            feat_u_bar, _, z_u_bar = model(x_u_bar)
            feat_l, pred_l, z_l = model(x_l)
            
            # 1. Labeled clustering with ground truth
            labeled_probs = feat2prob(z_l, labeled_centers)
            labeled_cluster_loss = F.cross_entropy(labeled_probs, y_l)
            
            # 2. Unlabeled clustering
            prob_u = feat2prob(z_u, model.encoder.center)
            prob_u_bar = feat2prob(z_u_bar, model.encoder.center)
            
            sharp_loss = F.kl_div(prob_u.log(), args.p_targets[idx_u].float().to(device))
            consistency_loss = F.mse_loss(prob_u, prob_u_bar)
            
            # 3. Separation between labeled and unlabeled centers
            center_separation = separation_loss(
                labeled_centers, 
                model.encoder.center, 
                margin=2.0
            )
            
            # 4. Repulsion: push unlabeled samples away from labeled centers
            repulsion = repulsion_loss(z_u, labeled_centers, temperature=0.1)
            
            # 5. Feature-level contrastive separation
            feature_separation = contrastive_separation_loss(feat_l, feat_u, temperature=0.5)
            
            # Combined loss
            loss = (labeled_cluster_loss + 
                   sharp_loss + 
                   consistency_loss + 
                   0.3 * center_separation + 
                   0.2 * repulsion + 
                   0.1 * feature_separation)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """
    Example usage of different methods
    """
    
    # Choose method
    method = "METHOD5_HYBRID"  # or METHOD1, METHOD2, METHOD3, METHOD4
    
    if method == "METHOD1":
        METHOD1_train(model, labeled_loader, unlabeled_loader, args)
    elif method == "METHOD2":
        METHOD2_train(model, labeled_loader, unlabeled_loader, args)
    elif method == "METHOD3":
        METHOD3_train(model, labeled_loader, unlabeled_loader, args)
    elif method == "METHOD4":
        METHOD4_train(model, labeled_loader, unlabeled_loader, args)
    elif method == "METHOD5_HYBRID":
        METHOD5_HYBRID_train(model, labeled_loader, unlabeled_loader, args)

if __name__ == "__main__":
    main()