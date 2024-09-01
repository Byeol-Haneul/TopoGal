import torch

class Invariants:
    def __init__(self, cell1, cell2):
        self.cell1 = cell1
        self.cell2 = cell2

    def cell_euclidean_distance(self):
        centroid1 = torch.tensor(self.cell1.centroid)
        centroid2 = torch.tensor(self.cell2.centroid)
        return torch.norm(centroid1 - centroid2)

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

        return hausdorff_distance

    def all_distances(self):
        distances = {
            'euclidean': self.cell_euclidean_distance(),
            'hausdorff': self.cell_hausdorff_distance()
        }
        return distances

def cell_invariants_torch(rank1_cells, rank2_cells):
    num_cells_rank1 = len(rank1_cells)
    num_cells_rank2 = len(rank2_cells)
    
    # Initialize tensors to store distances
    euclidean_distances = torch.zeros((num_cells_rank1, num_cells_rank2))
    hausdorff_distances = torch.zeros((num_cells_rank1, num_cells_rank2))

    # Compute distances between cells
    for i, cell1 in enumerate(rank1_cells):
        for j, cell2 in enumerate(rank2_cells):
            invariant = Invariants(cell1, cell2)
            euclidean_distance = invariant.cell_euclidean_distance()
            hausdorff_distance = invariant.cell_hausdorff_distance()
            
            # Assign distances to the respective tensors
            euclidean_distances[i, j] = euclidean_distance
            hausdorff_distances[i, j] = hausdorff_distance
    
    return euclidean_distances, hausdorff_distances
