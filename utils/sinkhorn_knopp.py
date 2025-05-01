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
        Applies Sinkhorn-Knopp normalization to logits.

        Steps:
        1. Exponentiate logits after temperature scaling.
        2. Normalize the matrix to sum to 1.
        3. Alternatingly normalize rows and columns.
        4. Return the balanced assignment matrix (soft labels).

        Args:
            logits (Tensor): Raw logits from the model [B, K]

        Returns:
            Tensor: Soft pseudo-labels [B, K]
        """

        # Step 1: Apply softmax-like transformation with temperature
        # Transpose so that rows represent clusters (K), columns represent samples (B)
        Q = torch.exp(logits / self.epsilon).t()  # Shape: [K, B]
        Q = Q + 1e-6 # avoid zero entries

        # Step 2: Normalize the entire matrix to sum to 1 (prevents scaling instability)
        Q /= Q.sum()  # Q is now a probability distribution over (K × B)

        B = Q.shape[1]  # Number of samples
        K = Q.shape[0]  # Number of clusters

        # Step 3: Sinkhorn iterations
        for it in range(self.num_iters):
            # Row normalization: each cluster (row) should sum to 1/K
            sum_of_rows = Q.sum(dim=1, keepdim=True)  # Shape: [K, 1]
            Q /= (sum_of_rows + 1e-6)
            Q /= K  # Ensures each row sums to 1/K

            # Column normalization: each sample (column) should sum to 1/B
            sum_of_cols = Q.sum(dim=0, keepdim=True)  # Shape: [1, B]
            Q /= (sum_of_cols +  + 1e-6)
            Q /= B  # Ensures each column sums to 1/B

        # Step 4: Rescale so each sample's distribution sums to 1
        Q *= B  # Now columns sum to 1

        # Step 5: Return back in [B, K] format (each row = soft pseudo-label for one sample)
        return Q.t()
