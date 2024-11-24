import torch
from torch.utils.data import Dataset
from utils.augmentation import augment_data

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, feature_names):
        self.feature_names = feature_names
        self.dataset = [
            {name: sample[i] for i, name in enumerate(self.feature_names)}
            for sample in data
        ]
        self.augmented_dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, augmented=False):
        return self.augmented_dataset[idx]

    def augment(self, drop_prob=0.1, cci_mode='euclidean'):
        self.augmented_dataset = [
            augment_data(sample, drop_prob, cci_mode) for sample in self.dataset
        ]


def pad_sparse_tensors(tensors):
    tensors = [t.to_dense() if t.is_sparse else t for t in tensors]
    max_size = tuple(max(s) for s in zip(*[t.size() for t in tensors]))
    padded_tensors = []
    for t in tensors:
        padded_tensor = torch.zeros(max_size, dtype=t.dtype)
        padded_tensor[:t.size(0), :t.size(1)] = t
        padded_tensors.append(padded_tensor.to_sparse())
    
    return torch.stack(padded_tensors)

def custom_collate_fn(batch):
    # Extract and pad tensors that require padding
    x_0_batch = pad_sparse_tensors([item['x_0'] for item in batch])
    x_1_batch = pad_sparse_tensors([item['x_1'] for item in batch])
    x_2_batch = pad_sparse_tensors([item['x_2'] for item in batch])
    x_3_batch = pad_sparse_tensors([item['x_3'] for item in batch])
    
    n0_to_0_batch = pad_sparse_tensors([item['n0_to_0'] for item in batch])
    n1_to_1_batch = pad_sparse_tensors([item['n1_to_1'] for item in batch])
    n2_to_2_batch = pad_sparse_tensors([item['n2_to_2'] for item in batch])
    n3_to_3_batch = pad_sparse_tensors([item['n3_to_3'] for item in batch])
    
    n0_to_1_batch = pad_sparse_tensors([item['n0_to_1'] for item in batch])
    n1_to_2_batch = pad_sparse_tensors([item['n1_to_2'] for item in batch])
    n2_to_3_batch = pad_sparse_tensors([item['n2_to_3'] for item in batch])

    global_feature_batch = torch.stack([item['global_feature'] for item in batch])
    y_batch = torch.stack([item['y'] for item in batch])

    return {
        'x_0': x_0_batch,
        'x_1': x_1_batch,
        'x_2': x_2_batch,
        'x_3': x_3_batch,
        'n0_to_0': n0_to_0_batch,
        'n1_to_1': n1_to_1_batch,
        'n2_to_2': n2_to_2_batch,
        'n3_to_3': n3_to_3_batch,
        'n0_to_1': n0_to_1_batch,
        'n1_to_2': n1_to_2_batch,
        'n2_to_3': n2_to_3_batch,
        'global_feature': global_feature_batch,
        'y': y_batch
    }
