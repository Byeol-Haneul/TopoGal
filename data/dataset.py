import torch
from torch.utils.data import Dataset

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


def collate_fn(data_batch):
    batched_data = {}
    
    for key in data_batch[0].keys():
        key_data = [item[key] for item in data_batch]

        # If the key starts with 'x_' or 'y_', treat it as N*F tensor (variable N, fixed F)
        if key.startswith('y') or key.startswith('global'):
            batched_data[key] = torch.stack(key_data, dim=0)
            continue
        elif key.startswith('x'):
            max_N = max([x.shape[0] for x in key_data])
            F = key_data[0].shape[1]  # F is fixed (number of features)

            # Pad each tensor along the N dimension (rows)
            padded_key_data = [torch.cat([x, torch.zeros(max_N - x.shape[0], F)], dim=0) for x in key_data]
            batched_data[key] = torch.stack(padded_key_data, dim=0)

        else:
            key_data = [key.to_dense() for key in key_data]
            max_M = max([n.shape[0] for n in key_data])
            max_N = max([n.shape[1] for n in key_data])

            # Pad each matrix to match max_M * max_N
            padded_key_data = [
                torch.cat([n, torch.zeros(max_M - n.shape[0], n.shape[1])], dim=0) if n.shape[0] < max_M else n
                for n in key_data
            ]
            padded_key_data = [
                torch.cat([n, torch.zeros(n.shape[0], max_N - n.shape[1])], dim=1) for n in padded_key_data
            ]
            
            # Stack them into a batch
            batched_data[key] = torch.stack(padded_key_data, dim=0)

    return batched_data
