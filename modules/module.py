import torch

# def feat2prob(feat, center, alpha=1.0):
#     q = 1.0 / (1.0 + torch.sum(
#         torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
#     q = q.pow((alpha + 1.0) / 2.0)
#     q = (q.t() / torch.sum(q, 1)).t()
#     return q


def feat2prob(feat, center, alpha=1.0):
    """
    Converts feature embeddings to probability distributions by computing similarity 
    between features and class centers.

    Args:
    - feat (torch.Tensor): Input features with shape [batch_size, n_dim]
    - center (torch.Tensor): Class centers with shape [n_clusters, n_dim]
    - alpha (float): Scaling parameter to control distribution sharpness.

    Returns:
    - q (torch.Tensor): Probability distribution over classes with shape [batch_size, n_cluster]
    """

    # Compute pairwise squared Euclidean distances between features and centers
    # feat.unsqueeze(1) -> shape [batch_size, 1, n_dim]
    # center -> shape [n_clusters, n_dim]
    # feat.unsqueeze(1) - center -> broadcasted to shape [batch_size, n_clusters, n_dim]
    diff = feat.unsqueeze(1) - center  # shape: [batch_size, n_clusters, n_dim]


    diff_square = torch.pow(diff, 2)  # shape: [batch_size, n_clusters, n_dim]
    
    #  Summing along the last dimension --> Distance from each cluster
    distances = torch.sum(diff_square, dim=2)  # shape: [batch_size, n_clusters]


    # Calculate q as the inverse distance measure, scaled by alpha
    # Adding 1 to distances for numerical stability, then dividing by alpha
    q = 1.0 / (1.0 + distances / alpha)  # shape: [batch_size, n_clusters]

    # Step 3: Raise q to the power of (alpha + 1) / 2 to control sharpness of distribution
    q = q.pow((alpha + 1.0) / 2.0)  # shape: [batch_size, n_clusters]

    # Normalize q to obtain a probability distribution over classes
    # q.t() / torch.sum(q, 1) -> first normalize across each row (class)
    # Then transpose back to get shape [batch_size, n_clusters]
    q = (q.t() / torch.sum(q, dim=1)).t()  # shape: [batch_size, n_clusters]

    return q


# def target_distribution(q):
#     weight = q**2 / q.sum(0)
#     return (weight.t() / weight.sum(1)).t()

def target_distribution(q):
    """
    Computes a target distribution for refining cluster assignments based on the 
    soft assignments (probability distribution) in q. The target distribution is 
    typically used to sharpen the probability distribution to focus on more 
    confident assignments.

    Args:
    - q (torch.Tensor): Soft assignment probabilities with shape [batch_size, n_clusters]

    Returns:
    - torch.Tensor: Refined target distribution with shape [batch_size, n_clusters]
    """
    
    #  Square the values in q to increase emphasis on high-confidence clusters. This makes clusters with higher probabilities more influential, sharpening the distribution.
    weight = q ** 2  # shape: [batch_size, n_clusters]

    # Normalize by the sum of q across all samples for each cluster.
    # Here, q.sum(0) computes the sum across the batch dimension (axis 0) for each cluster,
    # resulting in a vector of shape [n_clusters].
    # Dividing by q.sum(0) normalizes each cluster's contribution relative to its distribution across the batch,
    weight = weight / q.sum(0)  # shape: [batch_size, n_clusters]

    # Normalize each row to make a valid probability distribution.
    # Transpose weight to [n_clusters, batch_size], divide each element by the sum of its row (axis=1),
    # then transpose back to original shape [batch_size, n_clusters].
    # This row-wise normalization ensures that each sample’s distribution across clusters sums to 1.
    target_q = (weight.t() / weight.sum(1)).t()  # shape: [batch_size, n_clusters]

    return target_q

