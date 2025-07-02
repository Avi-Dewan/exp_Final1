import torch

class ClassAwareSinkhornKnopp:
    """
    Enhanced Sinkhorn-Knopp that adapts to class imbalance by using
    estimated class frequencies to guide the assignment process.
    """
    
    def __init__(self, num_iters=3, epsilon=0.05, momentum=0.9):
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Track class frequencies with momentum
        self.class_freq_estimate = None
        
    def update_class_frequencies(self, pseudo_labels):
        """Update running estimate of class frequencies"""
        with torch.no_grad():
            # Get soft class frequencies from current batch
            batch_freq = pseudo_labels.mean(dim=0)
            
            # Clamp to avoid extreme values
            batch_freq = torch.clamp(batch_freq, min=1e-6, max=1.0)
            
            if self.class_freq_estimate is None:
                self.class_freq_estimate = batch_freq.clone()
            else:
                self.class_freq_estimate = (self.momentum * self.class_freq_estimate + 
                                          (1 - self.momentum) * batch_freq)
                # Ensure frequencies stay reasonable
                self.class_freq_estimate = torch.clamp(self.class_freq_estimate, min=1e-6, max=1.0)
    
    def __call__(self, logits):
        """
        Enhanced Sinkhorn that considers class frequency estimates
        """
        with torch.no_grad():
            # Input stability - clamp extreme values
            logits = torch.clamp(logits, min=-50, max=50)
            
            # Subtract max for numerical stability before exp
            logits = logits - logits.max(dim=1, keepdim=True)[0]
            
            Q = torch.exp(logits / self.epsilon).t()  # Q is K x B
            B = Q.shape[1]  # batch size
            K = Q.shape[0]  # number of clusters
            
            # Add small epsilon to avoid zeros
            Q = Q + 1e-12
            
            # Standard Sinkhorn iterations with better stability
            for _ in range(self.num_iters):
                # Row normalization (sum over batch dimension)
                sum_Q = torch.sum(Q, dim=1, keepdim=True) + 1e-12
                Q = Q / sum_Q
                
                # Column normalization with class-aware adjustment
                if self.class_freq_estimate is not None and self.class_freq_estimate.device == logits.device:
                    # Use inverse frequency weighting for columns, but not too extreme
                    inv_freq = 1.0 / (self.class_freq_estimate + 1e-6)
                    # Clamp inverse frequencies to avoid extreme values
                    inv_freq = torch.clamp(inv_freq, min=0.1, max=10.0)
                    inv_freq = inv_freq / inv_freq.sum() * K  # normalize
                    target_col_sum = inv_freq.unsqueeze(1) * (B / K)
                else:
                    target_col_sum = B / K
                
                col_sum = torch.sum(Q, dim=0, keepdim=True) + 1e-12
                Q = Q * target_col_sum / col_sum
                
                # Ensure Q doesn't have extreme values
                Q = torch.clamp(Q, min=1e-12, max=1e12)
            
            # Final normalization to ensure valid probabilities
            pseudo_labels = Q.t()
            pseudo_labels = pseudo_labels / (pseudo_labels.sum(dim=1, keepdim=True) + 1e-12)
            
            # Update frequency estimate for next iteration
            self.update_class_frequencies(pseudo_labels)
            
            return pseudo_labels
