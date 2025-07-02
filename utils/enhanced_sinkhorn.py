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
        self.register_buffer = True
        
        # Track class frequencies with momentum
        self.class_freq_estimate = None
        
    def update_class_frequencies(self, pseudo_labels):
        """Update running estimate of class frequencies"""
        with torch.no_grad():
            # Get soft class frequencies from current batch
            batch_freq = pseudo_labels.mean(dim=0)
            
            if self.class_freq_estimate is None:
                self.class_freq_estimate = batch_freq.clone()
            else:
                self.class_freq_estimate = (self.momentum * self.class_freq_estimate + 
                                          (1 - self.momentum) * batch_freq)
    
    def __call__(self, logits):
        """
        Enhanced Sinkhorn that considers class frequency estimates
        """
        with torch.no_grad():
            Q = torch.exp(logits / self.epsilon).t()  # Q is K x B
            B = Q.shape[1]  # batch size
            K = Q.shape[0]  # number of clusters
            
            # Standard Sinkhorn iterations
            for _ in range(self.num_iters):
                # Row normalization (sum over batch dimension)
                sum_Q = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_Q
                
                # Column normalization with class-aware adjustment
                if self.class_freq_estimate is not None:
                    # Use inverse frequency weighting for columns
                    inv_freq = 1.0 / (self.class_freq_estimate + 1e-8)
                    inv_freq = inv_freq / inv_freq.sum() * K  # normalize
                    target_col_sum = inv_freq.unsqueeze(1) * (B / K)
                else:
                    target_col_sum = B / K
                
                Q *= target_col_sum / (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
            
            # Update frequency estimate for next iteration
            pseudo_labels = Q.t()
            self.update_class_frequencies(pseudo_labels)
            
            return pseudo_labels