import torch
import torch.nn as nn
import numpy as np

class SimCLR_Loss(nn.Module):
    """
    Implements the SimCLR loss function for contrastive learning using cosine similarity and cross-entropy loss.
    
    Parameters:
    - batch_size (int): The number of positive pairs in each batch.
    - temperature (float): A temperature scaling factor for the cosine similarities, controlling the concentration of the distribution.
    """
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        # Mask to filter out positive pairs (correlated samples)
        self.mask = self.mask_correlated_samples(batch_size)
         # Cross-entropy loss function to compute the contrastive loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
         # Cosine similarity function for pairwise similarity calculation
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """
        Creates a mask to exclude positive pairs from the set of negative samples.
        
        For a batch of size N, this mask is a (2N, 2N) boolean matrix where positive pairs 
        (corresponding augmented views) are set to 0, and all others to 1. This prevents the 
        positive pairs from being treated as negatives during loss calculation.
        
        Returns:
        - mask (torch.Tensor): A boolean mask of shape (2N, 2N).
        """
        N = 2 * batch_size # 2N total samples due to positive pairs
        mask = torch.ones((N, N), dtype=bool) # Start with all negatives ( mask = 1 )
        mask = mask.fill_diagonal_(0)  # Set diagonal to 0 to exclude self-similarities

        # Mark positive pairs (i, batch_size + i) and (batch_size + i, i) as 0
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Computes the SimCLR loss for a batch of embeddings.
        
        Parameters:
        - z_i (torch.Tensor): Embeddings from the first set of augmented views. [ batch_size, n_dim]
        - z_j (torch.Tensor): Embeddings from the second set of augmented views. [ batch_size, n_dim]
        
        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """

        N = 2 * self.batch_size # Total number of samples (pairs of augmented views)
        
        # Concatenate embeddings from both views to form a single batch : [2*batch_size, n_dim]
        z = torch.cat((z_i, z_j), dim=0) 

        # Compute pairwise cosine similarity for all samples, scaled by temperature: [batch_size, batch_size]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature 

         # Extract the positive similarities (diagonals) for both directions: 
        sim_i_j = torch.diag(sim, self.batch_size) # Positive Pair Similarity between z_i and z_j : [batch_size]
        sim_j_i = torch.diag(sim, -self.batch_size) # Positive Pair Similarity between z_j and z_i: [batch_size]
        
        # Combine positive similarities (z_i & z_j) into a single tensor (N, 1) for loss calculation: [2*batch_size, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # Extract negative samples using the mask and reshape to (N, -1) for loss calculation: [2*batch_size, n_dim - 2]
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        # Create labels: All positives are labeled 0, cross-entropy will treat them as correct class: [2*BatchSize]
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        # Concatenate positive and negative samples for cross-entropy computation: [2*batch_size, n_dim - 1]
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Compute the contrastive loss using cross-entropy. The first one is the postive one. The logits should predict that as well ( Cross Entorpy)
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by the batch size
        
        return loss