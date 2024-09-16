import torch
import torch.nn as nn
import sys

class PNAAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators=['mean', 'max'], scalers=['identity', 'amplification', 'attenuation'], delta=0.1):
        """
        PNA Aggregator for hypergraphs.
        
        Args:
            in_channels (int): Input feature dimension (D).
            out_channels (int): Output feature dimension (for the aggregated features).
            aggregators (list): List of aggregation functions ['mean', 'max', 'min', 'std'].
            scalers (list): List of scalers ['identity', 'amplification', 'attenuation'].
            delta (float): A scaling parameter for the amplification and attenuation functions.
        """
        super(PNAAggregator, self).__init__()
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta

        # Linear layer to transform the aggregated features
        self.mlp = nn.Linear(len(aggregators) * len(scalers) * in_channels, out_channels)

    def forward(self, neighborhood_matrix, node_features):
        """
        Forward pass of PNA aggregation.
        
        Args:
            neighborhood_matrix (torch.sparse_coo_tensor): MxN sparse matrix representing the neighborhood.
            node_features (torch.Tensor): NxD matrix where N is the number of nodes and D is the feature dimension.
        
        Returns:
            Aggregated features (MxD)
        """
        aggregated_features = []

        # Apply each aggregation method
        for agg in self.aggregators:
            agg_features = self.aggregate(neighborhood_matrix, node_features, agg)
            
            # Apply each scaling method
            for scaler in self.scalers:
                scaled_features = self.scale(agg_features, scaler, neighborhood_matrix)
                aggregated_features.append(scaled_features)

        # Concatenate all the aggregated features
        combined_aggregation = torch.cat(aggregated_features, dim=1)
        
        # Pass through a linear transformation to get the final output
        output = self.mlp(combined_aggregation)
        
        return output

    def aggregate(self, neighborhood_matrix, node_features, aggregation_type):
        """
        Apply aggregation (mean, max, min, std) over the neighborhood.
        """
        if aggregation_type == 'mean':
            return self.mean_aggregation(neighborhood_matrix, node_features)
        elif aggregation_type == 'max':
            return self.max_aggregation(neighborhood_matrix, node_features)
        elif aggregation_type == 'min':
            return self.min_aggregation(neighborhood_matrix, node_features)
        elif aggregation_type == 'std':
            return self.std_aggregation(neighborhood_matrix, node_features)
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    def mean_aggregation(self, neighborhood_matrix, node_features):
        """ Mean aggregation using sparse matrix multiplication. """
        return torch.sparse.mm(neighborhood_matrix, node_features)

    def max_aggregation(self, neighborhood_matrix, node_features):
        """Max aggregation using sparse matrix structure."""
        # Get the indices and values from the sparse matrix
        indices = neighborhood_matrix._indices()
        values = neighborhood_matrix._values()
        
        # Extract rows and columns from indices
        row, col = indices[0], indices[1]
        
        # Number of nodes and feature dimensions
        num_nodes = neighborhood_matrix.size(0)
        num_features = node_features.size(1)
        
        # Initialize max_features with zero tensors
        max_features = torch.zeros((num_nodes, num_features), device=node_features.device)
        
        # Keep track of the counts of contributions per node
        count = torch.zeros(num_nodes, device=node_features.device, dtype=torch.long)
        
        # Perform max aggregation
        for i in range(indices.size(1)):  # Loop over non-zero entries
            node_idx = row[i]
            feature_idx = col[i]
            max_features[node_idx] = torch.max(max_features[node_idx], node_features[feature_idx])
            count[node_idx] += 1
        
        # Handle nodes with no contributions by setting their features to zero
        max_features[count == 0] = 0
        
        return max_features

    def min_aggregation(self, neighborhood_matrix, node_features):
        """ Min aggregation using sparse matrix structure. """
        indices = neighborhood_matrix._indices()
        values = neighborhood_matrix._values()

        row, col = indices[0], indices[1]
        min_features = torch.full((neighborhood_matrix.size(0), node_features.size(1)), float('inf'), device=node_features.device)
        
        for i in range(values.size(0)):  # Loop over non-zero entries
            min_features[row[i]] = torch.min(min_features[row[i]], node_features[col[i]])

        return min_features

    def std_aggregation(self, neighborhood_matrix, node_features):
        """ Standard deviation aggregation based on mean. """
        mean_features = self.mean_aggregation(neighborhood_matrix, node_features)
        squared_diff = torch.zeros_like(mean_features)

        indices = neighborhood_matrix._indices()
        row, col = indices[0], indices[1]

        for i in range(row.size(0)):  # Loop over non-zero entries
            diff = node_features[col[i]] - mean_features[row[i]]
            squared_diff[row[i]] += diff ** 2
        
        count = torch.sparse.sum(neighborhood_matrix, dim=1).to_dense().unsqueeze(1).clamp(min=1)  # Avoid division by zero
        std_features = torch.sqrt(squared_diff / count)

        return std_features

    def scale(self, agg_features, scaler, neighborhood_matrix):
        """
        Scale the aggregated features using one of the scaling functions.
        """
        degrees = torch.sparse.sum(neighborhood_matrix, dim=1).to_dense().unsqueeze(1)  # Mx1
        
        if scaler == 'identity':
            return agg_features
        elif scaler == 'amplification':
            return agg_features * (degrees + self.delta)
        elif scaler == 'attenuation':
            return agg_features / (degrees + self.delta)
        else:
            raise ValueError(f"Unsupported scaler type: {scaler}")
