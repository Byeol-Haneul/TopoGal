import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config.param_config import PARAM_STATS, PARAM_ORDER


def normalize(tensor):
    return tensor
    '''
    if tensor.is_sparse:
        tensor = tensor.coalesce()
        values = tensor.values()
        max_val = values.max()
        normalized_values = values / max_val

        return torch.sparse_coo_tensor(
            tensor.indices(),
            normalized_values,
            tensor.size()
        )
    else:
        max_val = tensor.max()
        return tensor / max_val
    return tensor
    '''
    

def load_tensors(num_list, data_dir, label_filename, target_labels=None):
    label_file = pd.read_csv(label_filename, sep='\s+')

    y_list, x_0_list, x_1_list, x_2_list, x_3_list = [], [], [], [], []
    n0_to_0_list, n1_to_1_list, n2_to_2_list, n3_to_3_list = [], [], [], []
    n0_to_1_list, n1_to_2_list, n2_to_3_list, = [], [], []
    global_feature_list = []

    for num in tqdm(num_list):
        try:
            y = torch.Tensor(label_file.loc[num].to_numpy()[1:-1].astype(float))

            if target_labels:
                indices = [PARAM_ORDER.index(label) for label in target_labels]
                y = y[indices]

            x_0 = normalize(torch.load(os.path.join(data_dir, f"x_0_{num}.pt")))
            x_1 = normalize(torch.load(os.path.join(data_dir, f"x_1_{num}.pt")))
            x_2 = normalize(torch.load(os.path.join(data_dir, f"x_2_{num}.pt")))
            x_3 = normalize(torch.load(os.path.join(data_dir, f"x_3_{num}.pt")))
            
            n0_to_0 = normalize(torch.load(os.path.join(data_dir, f"n0_to_0_{num}.pt")))
            n1_to_1 = normalize(torch.load(os.path.join(data_dir, f"n1_to_1_{num}.pt")))
            n2_to_2 = normalize(torch.load(os.path.join(data_dir, f"n2_to_2_{num}.pt")))
            n3_to_3 = normalize(torch.load(os.path.join(data_dir, f"n3_to_3_{num}.pt")))

            n0_to_1 = normalize(torch.load(os.path.join(data_dir, f"n0_to_1_{num}.pt")))
            n1_to_2 = normalize(torch.load(os.path.join(data_dir, f"n1_to_2_{num}.pt")))
            n2_to_3 = normalize(torch.load(os.path.join(data_dir, f"n2_to_3_{num}.pt")))
            
            y_list.append(y)
            x_0_list.append(x_0)
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            x_3_list.append(x_3)
            n0_to_0_list.append(n0_to_0)
            n1_to_1_list.append(n1_to_1)
            n2_to_2_list.append(n2_to_2)
            n3_to_3_list.append(n3_to_3)
            n0_to_1_list.append(n0_to_1)
            n1_to_2_list.append(n1_to_2)
            n2_to_3_list.append(n2_to_3)

            # Calculate global features based on the shape of x_0, x_1, x_2, x_3
            global_feature = torch.tensor([
                x_0.shape[0],
                x_1.shape[0],
                x_2.shape[0],
                x_3.shape[0]
            ], dtype=torch.float32).unsqueeze(0)  # Shape [1, 4]

            global_feature = torch.log10(global_feature+1)
            global_feature_list.append(global_feature)


        except Exception as e:
            print(f"Error loading tensors for num {num}: {e}")

    return (
        y_list, x_0_list, x_1_list, x_2_list, x_3_list,
        n0_to_0_list, n1_to_1_list, n2_to_2_list, n3_to_3_list, 
        n0_to_1_list, n1_to_2_list, n2_to_3_list,
        global_feature_list
    )


def split_data(*lists, test_size=0.15, val_size=0.15):
    train_val_test_lists = []
    for lst in lists:
        train, temp = train_test_split(lst, test_size=test_size + val_size, shuffle=False)
        val, test = train_test_split(temp, test_size=test_size / (test_size + val_size), shuffle=False)
        train_val_test_lists.append((train, val, test))
    return train_val_test_lists
