import torch
import sys
import cProfile
import pstats
from config_preprocess import *

eps = 1e-20

class Invariants:
    def __init__(self, cell1, cell2):
        self.cell1 = cell1
        self.cell2 = cell2

    def cell_euclidean_distance(self):
        centroid1 = torch.tensor(self.cell1.centroid)
        centroid2 = torch.tensor(self.cell2.centroid)
        return torch.norm(centroid1 - centroid2) + eps

    def cell_hausdorff_distance(self):
        node_pos1 = torch.tensor(self.cell1.node_position)
        node_pos2 = torch.tensor(self.cell2.node_position)

        # Compute all pairwise distances between points in the two sets
        pairwise_distances = torch.cdist(node_pos1, node_pos2)

        # Calculate the directed Hausdorff distances
        hausdorff_dist_1 = torch.max(torch.min(pairwise_distances, dim=1)[0])
        hausdorff_dist_2 = torch.max(torch.min(pairwise_distances, dim=0)[0])

        # The Hausdorff distance is the maximum of the directed distances
        hausdorff_distance = torch.max(hausdorff_dist_1, hausdorff_dist_2)

        return hausdorff_distance + eps

    def all_distances(self):
        distances = {
            'euclidean': self.cell_euclidean_distance(),
            'hausdorff': self.cell_hausdorff_distance()
        }
        return distances

def cell_invariants_torch(rank1_cells, rank2_cells, neighbor_matrix):
    num_cells_rank1 = len(rank1_cells)
    num_cells_rank2 = len(rank2_cells)
    
    # Initialize tensors to store distances
    euclidean_distances = torch.zeros((num_cells_rank1, num_cells_rank2))
    hausdorff_distances = torch.zeros((num_cells_rank1, num_cells_rank2))

    # Access the indices of non-zero elements in the sparse matrix
    indices = neighbor_matrix.indices().T.numpy()

    # Iterate through the non-zero entries using the indices
    for i, j in indices:      
        cell1 = rank1_cells[i]
        cell2 = rank2_cells[j]
                    
        invariant = Invariants(cell1, cell2)
        euclidean_distance = invariant.cell_euclidean_distance()
        hausdorff_distance = invariant.cell_hausdorff_distance()
        
        # Assign distances to the respective tensors
        euclidean_distances[i, j] = euclidean_distance
        hausdorff_distances[i, j] = hausdorff_distance

    return euclidean_distances.to_sparse(), hausdorff_distances.to_sparse()


# Wrapper function to handle profiling
def profile_if_enabled(func):
    def wrapper(*args, **kwargs):
        if ENABLE_PROFILING:
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)  # Print the top 20 results, sorted by cumulative time
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

cell_invariants_torch = profile_if_enabled(cell_invariants_torch)

def cross_cell_invariants(num, nodes, edges, tetrahedra, clusters, hyperclusters, neighbors):
    cell_lists = [nodes, edges, tetrahedra, list(clusters.values()), list(hyperclusters.values())]
    rank_names = ['0', '1', '2', '3', '4']

    invariants = {}

    for i, list1 in enumerate(cell_lists):
        for j, list2 in enumerate(cell_lists):
            if i <= j:
                print(f"[LOG] Calculating for cell ranks {i} and {j}", file=sys.stderr)
                # Compute invariants using the cell_invariants_torch function
                euclidean_distances, hausdorff_distances = cell_invariants_torch(list1, list2, neighbors[f"n{i}_to_{j}"])

                # Store the results in the dictionary
                invariants[f'euclidean_{rank_names[i]}_to_{rank_names[j]}'] = normalize(euclidean_distances, "ISDISTANCE")
                invariants[f'hausdorff_{rank_names[i]}_to_{rank_names[j]}'] = normalize(hausdorff_distances, "ISDISTANCE")


    for key, tensor in invariants.items():
        print(f"[LOG] Saving tensor {key}.pt", file=sys.stderr)
        torch.save(tensor, os.path.join(tensor_dir, f"{key}_{num}.pt"))

    return invariants