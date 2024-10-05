from mpi4py import MPI
import torch
import numpy as np
from toponetx.readwrite.serialization import load_from_pickle
import pandas as pd
import os
import sys
from config_preprocess import *
import scipy 

def get_neighbors(num, cc):
    in_channels = [-1, -1, -1, -1, -1]

    results = {}
    label_file = pd.read_csv(label_filename, sep='\s+')
    
    try:
        in_filename = "data_" + str(num) + ".pickle"
        print(f"[LOG] Loading pickle file {in_filename}", file=sys.stderr)

        if cc == None:
            cc = load_from_pickle(cc_dir + in_filename)
        
        print(f"[LOG] Processing node features for num {num}", file=sys.stderr)
        x_0 = list(cc.get_node_attributes("node_feat").values())
        in_channels[0] = len(x_0[0])
        x_0 = torch.tensor(np.stack(x_0)).reshape(-1, in_channels[0])
        results['x_0'] = x_0
    
        print(f"[LOG] Processing edge features for num {num}", file=sys.stderr)
        x_1 = list(cc.get_cell_attributes("edge_feat").values())
        in_channels[1] = len(x_1[0])
        x_1 = torch.tensor(np.stack(x_1)).reshape(-1, in_channels[1])
        results['x_1'] = x_1
    
        print(f"[LOG] Processing tetra features for num {num}", file=sys.stderr)
        x_2 = list(cc.get_cell_attributes("tetra_feat").values())
        in_channels[2] = len(x_2[0])
        x_2 = torch.tensor(np.stack(x_2)).reshape(-1, in_channels[2])
        results['x_2'] = x_2
    
        print(f"[LOG] Processing cluster features for num {num}", file=sys.stderr)
        x_3 = list(cc.get_cell_attributes("cluster_feat").values())
        in_channels[3] = len(x_3[0])
        x_3 = torch.tensor(np.stack(x_3)).reshape(-1, in_channels[3])
        results['x_3'] = x_3

        print(f"[LOG] Processing cluster features for num {num}", file=sys.stderr)
        x_4 = list(cc.get_cell_attributes("hypercluster_feat").values())
        in_channels[4] = len(x_4[0])
        x_4 = torch.tensor(np.stack(x_4)).reshape(-1, in_channels[4])
        results['x_4'] = x_4
        
        
        print(f"[LOG] Processing adjacency and incidence matrices for num {num}", file=sys.stderr)
        
        # ADJACENCY
        print(f"[LOG] Computing n0_to_0 for num {num}", file=sys.stderr)
        n0_to_0 = cc.adjacency_matrix(rank=0, via_rank=1)  # Nodes in the same cluster
        n0_to_0 += scipy.sparse.eye(n0_to_0.shape[0])
        results['n0_to_0'] = torch.from_numpy(n0_to_0.todense())

        print(f"[LOG] Computing n1_to_1 for num {num}", file=sys.stderr)
        n1_to_1 = cc.adjacency_matrix(rank=1, via_rank=2)
        n1_to_1 += scipy.sparse.eye(n1_to_1.shape[0])  # Adding identity matrix
        results['n1_to_1'] = torch.from_numpy(n1_to_1.todense())

        print(f"[LOG] Computing n2_to_2 (adjacency) for num {num}", file=sys.stderr)
        n2_to_2 = cc.adjacency_matrix(rank=2, via_rank=3)
        n2_to_2 += scipy.sparse.eye(n2_to_2.shape[0])  # Adding identity matrix
        results['n2_to_2'] = torch.from_numpy(n2_to_2.todense())

        print(f"[LOG] Computing n3_to_3 (adjacency) for num {num}", file=sys.stderr)
        n3_to_3 = cc.adjacency_matrix(rank=3, via_rank=4)  # Clusters sharing edges
        n3_to_3 += scipy.sparse.eye(n3_to_3.shape[0])  # Adding identity matrix
        results['n3_to_3'] = torch.from_numpy(n3_to_3.todense())

        print(f"[LOG] Computing n4_to_4 (coadjacency) for num {num}", file=sys.stderr)
        n4_to_4 = cc.coadjacency_matrix(rank=4, via_rank=3)  # Clusters sharing edges
        n4_to_4 += scipy.sparse.eye(n4_to_4.shape[0])  # Adding identity matrix
        results['n4_to_4'] = torch.from_numpy(n4_to_4.todense())


        ## INCIDENCE
        print(f"[LOG] Computing n0_to_1 for num {num}", file=sys.stderr)
        n0_to_1 = cc.incidence_matrix(rank=0, to_rank=1)
        results['n0_to_1'] = torch.from_numpy(n0_to_1.todense())

        print(f"[LOG] Computing n0_to_2 for num {num}", file=sys.stderr)
        n0_to_2 = cc.incidence_matrix(rank=0, to_rank=2)
        results['n0_to_2'] = torch.from_numpy(n0_to_2.todense())
        
        print(f"[LOG] Computing n0_to_3 for num {num}", file=sys.stderr)
        n0_to_3 = cc.incidence_matrix(rank=0, to_rank=3)
        results['n0_to_3'] = torch.from_numpy(n0_to_3.todense())

        print(f"[LOG] Computing n0_to_4 for num {num}", file=sys.stderr)
        n0_to_4 = cc.incidence_matrix(rank=0, to_rank=4)
        results['n0_to_4'] = torch.from_numpy(n0_to_4.todense())

        print(f"[LOG] Computing n1_to_2 for num {num}", file=sys.stderr)
        n1_to_2 = cc.incidence_matrix(rank=1, to_rank=2)
        results['n1_to_2'] = torch.from_numpy(n1_to_2.todense())
        
        print(f"[LOG] Computing n1_to_3 for num {num}", file=sys.stderr)
        n1_to_3 = cc.incidence_matrix(rank=1, to_rank=3)
        results['n1_to_3'] = torch.from_numpy(n1_to_3.todense())

        print(f"[LOG] Computing n1_to_4 for num {num}", file=sys.stderr)
        n1_to_4 = cc.incidence_matrix(rank=1, to_rank=4)
        results['n1_to_4'] = torch.from_numpy(n1_to_4.todense())

        print(f"[LOG] Computing n2_to_3 for num {num}", file=sys.stderr)
        n2_to_3 = cc.incidence_matrix(rank=2, to_rank=3)
        results['n2_to_3'] = torch.from_numpy(n2_to_3.todense())

        print(f"[LOG] Computing n2_to_4 for num {num}", file=sys.stderr)
        n2_to_4 = cc.incidence_matrix(rank=2, to_rank=4)
        results['n2_to_4'] = torch.from_numpy(n2_to_4.todense())

        print(f"[LOG] Computing n3_to_4 for num {num}", file=sys.stderr)
        n3_to_4 = cc.incidence_matrix(rank=3, to_rank=4)
        results['n3_to_4'] = torch.from_numpy(n3_to_4.todense())
        
        print(f"[LOG] Global feature for num {num}", file=sys.stderr)
        feature_list = [results[f'x_{i}'].shape[0] for i in range(5)]
        global_feature = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)  # Shape [1, 5]
        global_feature = torch.log10(global_feature + 1)
        results['global_feature'] = global_feature

        print(f"[LOG] Global feature for num {num}", file=sys.stderr)
        feature_list = [results[f'x_{i}'].shape[0] for i in range(5)]
        global_feature = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)  # Shape [1, 5]
        global_feature = torch.log10(global_feature + 1)
        results['global_feature'] = global_feature

        # Save results
        for key, tensor in results.items():
            print(f"[LOG] Saving tensor {key}_{num}.pt", file=sys.stderr)
            torch.save(tensor, os.path.join(tensor_dir, f"{key}_{num}.pt"))

    except Exception as e:
        print(f"[LOG] Error processing num {num}: {e}", file=sys.stderr)

    return results


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[LOG] MPI initialized with rank {rank} and size {size}", file=sys.stderr)

    # Determine the range of numbers to process for this rank
    total_nums = 1000  # For example, if you have 10 datasets
    nums_per_rank = total_nums // size
    start_num = rank * nums_per_rank
    end_num = start_num + nums_per_rank

    # If total_nums is not perfectly divisible by size, the last rank handles the remainder
    if rank == size - 1:
        end_num = total_nums

    print(f"[LOG] Rank {rank} processing range {start_num} to {end_num}", file=sys.stderr)

    # Process the numbers assigned to this rank
    for num in range(start_num, end_num):
        print(f"[LOG] Rank {rank} processing num {num}", file=sys.stderr)
        get_neighbors(num, None)

    print(f"[LOG] Rank {rank} completed processing", file=sys.stderr)

if __name__ == "__main__":
    main()
