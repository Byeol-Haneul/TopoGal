import sys
import os

import torch
from config_preprocess import *

def padding(feature):
    tensor_list = []
    
    # Load tensors
    for num in range(1000):
        tensor_path = os.path.join(tensor_dir, f"{feature}_{num}.pt")
        tensor = torch.load(tensor_path)
        tensor_list.append(tensor)
    
    # Pad dense or sparse tensors depending on feature
    if feature.startswith('y') or feature.startswith('global'):
        # No padding required, skipping
        padded_tensors = tensor_list
    elif feature.startswith('x'):  # Variable-length dense tensors (N * F)
        max_N = max([t.shape[0] for t in tensor_list])
        F = tensor_list[0].shape[1]  # Number of features is fixed
        
        # Pad each tensor to match max_N dimension
        padded_tensors = [
            torch.cat([t, torch.zeros(max_N - t.shape[0], F)], dim=0) for t in tensor_list
        ]
    else:  # Assuming sparse tensor padding (non-dense tensors)
        max_M = max([n.shape[0] for n in tensor_list])
        max_N = max([n.shape[1] for n in tensor_list])

        padded_tensors = []
        for sparse_tensor in tensor_list:
            new_sparse_tensor = torch.sparse_coo_tensor(
                sparse_tensor.indices(),  
                sparse_tensor.values(),   
                size=(max_M, max_N),     
                dtype=sparse_tensor.dtype,
                device=sparse_tensor.device
            )
            padded_tensors.append(new_sparse_tensor)
    
    # Save padded tensors to output directory
    for num, padded_tensor in enumerate(padded_tensors):
        output_path = os.path.join(pad_dir, f"{feature}_{num}.pt")
        torch.save(padded_tensor, output_path)
    
    print(f"[LOG] Padded tensors for {feature} saved to {pad_dir}", file=sys.stderr)

if __name__ == "__main__":
    for feature in feature_sets:
        print(f"[LOG] Padding for {feature} in dir {tensor_dir}", file=sys.stderr)
        padding(feature)