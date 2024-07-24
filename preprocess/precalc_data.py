from mpi4py import MPI
import torch
import numpy as np
from toponetx.readwrite.serialization import load_from_pickle
import pandas as pd
import os

# Define your in_channels and file paths
in_channels = [5, 1, 1, 1]
in_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/cc/"
label_filename = "/data2/jylee/topology/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
output_save_dir = "/data2/jylee/topology/IllustrisTNG/combinatorial/tensors/"

# Create the directory if it doesn't exist
os.makedirs(output_save_dir, exist_ok=True)

def process_num(num):
    results = {}
    label_file = pd.read_csv(label_filename, delim_whitespace=True)
    
    try:
        in_filename = "data_" + str(num) + ".pickle"
        cc = load_from_pickle(in_dir + in_filename)
        
        x_0 = list(cc.get_node_attributes("node_feat").values())
        x_0 = torch.tensor(np.stack(x_0)).reshape(-1, in_channels[0])
        results['x_0'] = x_0
    
        x_1 = list(cc.get_cell_attributes("edge_feat").values())
        x_1 = torch.tensor(np.stack(x_1)).reshape(-1, in_channels[1])
        results['x_1'] = x_1
    
        x_2 = list(cc.get_cell_attributes("tetra_feat").values())
        x_2 = torch.tensor(np.stack(x_2)).reshape(-1, in_channels[2])
        results['x_2'] = x_2
    
        x_3 = list(cc.get_cell_attributes("cluster_feat").values())
        x_3 = torch.tensor(np.stack(x_3)).reshape(-1, in_channels[3])
        results['x_3'] = x_3
        
        #results['y'] = torch.tensor(label_file.loc[num].to_numpy()[1:].astype(float))

        # Compute adjacency and incidence matrices
        n0_to_0 = cc.adjacency_matrix(rank=0, via_rank=1)
        results['n0_to_0'] = torch.from_numpy(n0_to_0.todense()).to_sparse()

        n1_to_1 = cc.adjacency_matrix(rank=1, via_rank=2)
        results['n1_to_1'] = torch.from_numpy(n1_to_1.todense()).to_sparse()

        n2_to_2 = cc.coadjacency_matrix(rank=2, via_rank=1) #coadj
        results['n2_to_2'] = torch.from_numpy(n2_to_2.todense()).to_sparse()

        n0_to_1 = cc.incidence_matrix(rank=0, to_rank=1)
        results['n0_to_1'] = torch.from_numpy(n0_to_1.todense()).to_sparse()

        n1_to_2 = cc.incidence_matrix(rank=1, to_rank=2)
        results['n1_to_2'] = torch.from_numpy(n1_to_2.todense()).to_sparse()
        
        # Save results
        for key, tensor in results.items():
            torch.save(tensor, os.path.join(output_save_dir, f"{key}_{num}.pt"))

    except Exception as e:
        print(f"Error processing num {num}: {e}")

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the range of numbers to process for this rank
    total_nums = 1000  # For example, if you have 10 datasets
    nums_per_rank = total_nums // size
    start_num = rank * nums_per_rank
    end_num = start_num + nums_per_rank

    # If total_nums is not perfectly divisible by size, the last rank handles the remainder
    if rank == size - 1:
        end_num = total_nums

    # Process the numbers assigned to this rank
    for num in range(start_num, end_num):
        process_num(num)

if __name__ == "__main__":
    main()
