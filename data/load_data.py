import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_tensors(num_list, output_save_dir, label_filename):
    label_file = pd.read_csv(label_filename, delim_whitespace=True)

    y_list, x_0_list, x_1_list, x_2_list, x_3_list = [], [], [], [], []
    n0_to_0_list, n1_to_1_list, n2_to_2_list, n0_to_1_list, n1_to_2_list = [], [], [], [], []

    for num in tqdm(num_list):
        try:
            x_0 = torch.load(os.path.join(output_save_dir, f"x_0_{num}.pt"))
            x_1 = torch.load(os.path.join(output_save_dir, f"x_1_{num}.pt"))
            x_2 = torch.load(os.path.join(output_save_dir, f"x_2_{num}.pt"))
            x_3 = torch.load(os.path.join(output_save_dir, f"x_3_{num}.pt"))
            
            n0_to_0 = torch.load(os.path.join(output_save_dir, f"n0_to_0_{num}.pt"))
            n1_to_1 = torch.load(os.path.join(output_save_dir, f"n1_to_1_{num}.pt"))
            n2_to_2 = torch.load(os.path.join(output_save_dir, f"n2_to_2_{num}.pt"))
            n0_to_1 = torch.load(os.path.join(output_save_dir, f"n0_to_1_{num}.pt"))
            n1_to_2 = torch.load(os.path.join(output_save_dir, f"n1_to_2_{num}.pt"))

            y = torch.Tensor(label_file.loc[num].to_numpy()[1:-1].astype(float))
            
            x_0_list.append(x_0)
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            x_3_list.append(x_3)
            y_list.append(y)
            n0_to_0_list.append(n0_to_0)
            n1_to_1_list.append(n1_to_1)
            n2_to_2_list.append(n2_to_2)
            n0_to_1_list.append(n0_to_1)
            n1_to_2_list.append(n1_to_2)

        except Exception as e:
            print(f"Error loading tensors for num {num}: {e}")

    return (
        y_list, x_0_list, x_1_list, x_2_list, x_3_list,
        n0_to_0_list, n1_to_1_list, n2_to_2_list, n0_to_1_list, n1_to_2_list
    )

def split_data(*lists, test_size=0.3):
    train_test_lists = []
    for lst in lists:
        train, test = train_test_split(lst, test_size=test_size, shuffle=False)
        train_test_lists.append((train, test))
    return train_test_lists
