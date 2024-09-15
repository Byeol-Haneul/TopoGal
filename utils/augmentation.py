import torch


### CURRENTLY VERY SLOW ###

def generate_mask(sparse_tensor, drop_prob=0.1):
    """
    Generate a sparse mask tensor for the given sparse tensor based on the drop probability.

    Args:
    - sparse_tensor (torch.sparse_coo_tensor): Sparse tensor to generate a mask for.
    - drop_prob (float): Probability of dropping a connection.
    
    Returns:
    - mask (torch.sparse_coo_tensor): Sparse tensor mask indicating which values to keep.
    """
    values = sparse_tensor.values()
    mask_values = torch.rand(values.shape, dtype=torch.float) > drop_prob
    mask_indices = sparse_tensor.indices()
    mask = torch.sparse_coo_tensor(mask_indices, mask_values, sparse_tensor.size()).coalesce()
    return mask


def augment_matrix(matrix, mask):
    """
    Augment a sparse matrix by applying the given mask to its non-zero values.

    Args:
    - matrix (torch.sparse_coo_tensor): Sparse COO matrix.
    - mask (torch.sparse_coo_tensor): Sparse mask tensor.

    Returns:
    - Augmented sparse COO matrix.
    """
    # Apply the mask to the matrix using sparse_mask
    augmented_matrix = matrix.sparse_mask(mask)
    
    # Filter out zeroed values
    return augmented_matrix

def augment_batch(batch, drop_prob=0.1):
    """
    Augments a dictionary of neighborhood matrices and cci matrices by making some nonzero elements zero.

    Args:
    - batch (dict): Dictionary where keys are matrix names and values are sparse COO matrices.
    - drop_prob (float): Probability of dropping a connection.
    
    Returns:
    - A new dictionary with augmented neighborhood and cci matrices.
    """
    neighborhood_keys = [
        'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
        'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
        'n1_to_2', 'n1_to_3', 'n1_to_4',
        'n2_to_3', 'n2_to_4',
        'n3_to_4'
    ]
    
    cci_keys = [
        'euclidean_0_to_0', 'euclidean_1_to_1', 'euclidean_2_to_2', 'euclidean_3_to_3', 'euclidean_4_to_4',
        'euclidean_0_to_1', 'euclidean_0_to_2', 'euclidean_0_to_3', 'euclidean_0_to_4',
        'euclidean_1_to_2', 'euclidean_1_to_3', 'euclidean_1_to_4',
        'euclidean_2_to_3', 'euclidean_2_to_4',
        'euclidean_3_to_4',
    ]

    # Create a mapping of neighborhood keys to cci keys
    key_mapping = dict(zip(neighborhood_keys, cci_keys))
    
    augmented_dict = {}

    for neigh_key, cci_key in key_mapping.items():
        if neigh_key in batch and cci_key in batch:
            mask = generate_mask(batch[neigh_key], drop_prob)
            augmented_dict[neigh_key] = augment_matrix(batch[neigh_key], mask)
            augmented_dict[cci_key] = augment_matrix(batch[cci_key], mask)
        elif cci_key in batch:
            # If no matching neighborhood key, keep cci matrix unchanged
            augmented_dict[cci_key] = batch[cci_key]

    for key in batch:
        if key not in neighborhood_keys and key not in cci_keys:
            augmented_dict[key] = batch[key]
    
    return augmented_dict
