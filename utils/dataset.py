import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y = sample[0]
        x_0, x_1, x_2, x_3 = sample[1:5]
        n0_to_0, n1_to_1, n2_to_2, n0_to_1, n1_to_2 = sample[5:]
        
        return {
            'y': y,
            'x_0': x_0, 'x_1': x_1, 'x_2': x_2, 'x_3': x_3,
            'n0_to_0': n0_to_0, 'n1_to_1': n1_to_1, 'n2_to_2': n2_to_2,
            'n0_to_1': n0_to_1, 'n1_to_2': n1_to_2
        }
