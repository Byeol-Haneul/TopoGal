import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import time 

class CustomDataset(Dataset):
    def __init__(self, data, feature_names):
        """
        Args:
            data (list of tuples): List of samples where each sample is a tuple containing all features and target.
            feature_names (list of str): List of feature names in the order they appear in the tuples.
        """
        self.data = data
        self.feature_names = feature_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Dynamically construct the feature dictionary
        feature_dict = {name: sample[i] for i, name in enumerate(self.feature_names)}
        return feature_dict


def timeit_wrapper(func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds")
        return result
    return timed

'''
@timeit_wrapper
def collate_fn(data_batch):
    batched_data = {}
    for key in data_batch[0].keys():
        batched_data[key] = torch.stack([item[key].to_dense() for item in data_batch], dim=0)
    return batched_data
'''
@timeit_wrapper
def collate_fn(data_batch):
    batched_data = {}
    
    for key in data_batch[0].keys():
        key_data = [item[key] for item in data_batch]

        # If the key starts with 'x_' or 'y_', treat it as N*F tensor (variable N, fixed F)
        if key.startswith('y') or key.startswith('global'):
            batched_data[key] = torch.stack(key_data, dim=0)
            continue
        elif key.startswith('x'):
            batched_data[key] = pad_sequence(key_data, batch_first=True)

        else:
            max_M = max([n.shape[0] for n in key_data])
            max_N = max([n.shape[1] for n in key_data])

            padded_key_data = []
            for sparse_tensor in key_data:
                new_sparse_tensor = torch.sparse_coo_tensor(
                    sparse_tensor.indices(),  
                    sparse_tensor.values(),   
                    size=(max_M, max_N),     
                    dtype=sparse_tensor.dtype,
                    device=sparse_tensor.device
                )
                padded_key_data.append(new_sparse_tensor.to_dense())

            batched_data[key] = torch.stack(padded_key_data, dim=0)

    return batched_data
