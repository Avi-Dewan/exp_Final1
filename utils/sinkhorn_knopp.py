import torch


class SinkhornKnopp(torch.nn.Module):
    """
    Implements the Sinkhorn-Knopp algorithm for generating balanced soft pseudo-labels
    from clustering logits. This is used in self-supervised / unsupervised learning
    (e.g., SwAV, UNO) to avoid degenerate cluster assignments.

    The output is a doubly-stochastic matrix (rows and columns sum to constants), which
    represents a balanced soft assignment of each sample to clusters.

    Args:
        num_iters (int): Number of row/column normalization iterations.
        epsilon (float): Temperature for softmax scaling (lower = sharper).

    Input:
        logits (Tensor): Raw logits (before softmax) of shape (B, K), where
                         - B = batch size
                         - K = number of clusters (prototypes)

    Output:
        Tensor: Soft pseudo-labels of shape (B, K), where each row is a probability
                distribution summing to 1, and the global assignment is approximately balanced
                (i.e., each cluster gets ~B/K total mass).

    Example usage:
        sk = SinkhornKnopp(num_iters=3, epsilon=0.05)
        soft_labels = sk(logits)  # logits: shape [B, K]
    """

    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
 
    @torch.no_grad()
    def forward(self, logits):
        """
        Applies numerically stable Sinkhorn-Knopp normalization to logits.

        Args:
            logits (Tensor): Raw logits from the model [B, K]

        Returns:
            Tensor: Soft pseudo-labels [B, K]
        """
        # Step 0: Stabilize logits before exponentiation
        logits = logits - logits.max(dim=1, keepdim=True)[0]  # [B, K]

        # Step 1: Softmax-like exponentiation with temperature scaling
        Q = torch.exp(logits / self.epsilon).t()  # [K, B]
        Q += 1e-6  # Avoid exact zero

        # Step 2: Normalize total mass to 1
        Q /= Q.sum()

        B = Q.shape[1]  # #samples
        K = Q.shape[0]  # #clusters

        # Step 3: Sinkhorn iterations (row and column normalization)
        for _ in range(self.num_iters):
            Q /= (Q.sum(dim=1, keepdim=True) + 1e-6)  # Row norm
            Q /= K
            Q /= (Q.sum(dim=0, keepdim=True) + 1e-6)  # Column norm
            Q /= B

        Q *= B  # Rescale: columns should sum to 1
        return Q.t()  # [B, K]
