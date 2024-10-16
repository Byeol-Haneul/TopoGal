import torch
import torch.nn as nn
import sys
from typing import Literal

def default_agg(neighbor, x):
    return torch.mm(neighbor, x)

class PNAAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators=['mean', 'max', 'min', 'std'], scalers=['identity']):
        super(PNAAggregator, self).__init__()
        self.aggregators = aggregators
        self.scalers = scalers
        self.mlp = nn.Linear(len(aggregators) * len(scalers) * in_channels, out_channels)
        self.activation = nn.Tanh()

    def forward(self, neighborhood_matrix, features):
        self.sparse = neighborhood_matrix
        self.features = features

        dense = neighborhood_matrix.to_dense()
        self.product = torch.einsum('mn,nk->mnk', dense, features)
        degrees = dense.sum(dim=1)
        delta = torch.mean(torch.log10(degrees+1))
        self.S_amp = (degrees/delta).unsqueeze(1)

        aggregated_features = []
        for agg in self.aggregators:
            agg_features = self.aggregate(agg)
            
            for scaler in self.scalers:
                scaled_features = self.scale(agg_features, scaler)
                aggregated_features.append(scaled_features)

        combined_aggregation = torch.cat(aggregated_features, dim=1)
        output = self.mlp(combined_aggregation)
        output = self.activation(output)
        return output

    def aggregate(self, aggregation_type):
        if aggregation_type == 'mean':
            return self.mean_aggregation()
        elif aggregation_type == 'max':
            return self.max_aggregation()
        elif aggregation_type == 'min':
            return self.min_aggregation()
        elif aggregation_type == 'std':
            return self.std_aggregation()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    def mean_aggregation(self):
        return self.product.mean(dim=1)

    def max_aggregation(self):
        return self.product.max(dim=1)[0]

    def min_aggregation(self):
        return self.product.min(dim=1)[0]

    def std_aggregation(self):
        return self.product.std(dim=1)

    def scale(self, agg_features, scaler):
        if scaler == 'identity':
            return agg_features
        elif scaler == 'amplification':
            return agg_features * self.S_amp
        elif scaler == 'attenuation':
            return agg_features / self.S_amp
        else:
            raise ValueError(f"Unsupported scaler type: {scaler}")

class RankAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators=['mean', 'max', 'min', 'std']):
        super(RankAggregator, self).__init__()
        self.aggregators = aggregators

        # Linear layer to transform the aggregated features
        self.mlp = nn.Linear(len(aggregators) * in_channels, out_channels)
        self.activation = nn.Tanh()

    def forward(self, node_features_list):
        aggregated_features = []
        self.num_features = node_features_list[0].size(1)  # Dimension D of features

        # Apply each aggregation method
        for agg in self.aggregators:
            agg_features = self.aggregate(node_features_list, agg)
            aggregated_features.append(agg_features)

        # Concatenate all the aggregated features
        combined_aggregation = torch.cat(aggregated_features, dim=-1)
        output = self.mlp(combined_aggregation)
        output = self.activation(output)
        return output

    def aggregate(self, node_features_list, aggregation_type):
        if aggregation_type == 'mean':
            return self.mean_aggregation(node_features_list)
        elif aggregation_type == 'max':
            return self.max_aggregation(node_features_list)
        elif aggregation_type == 'min':
            return self.min_aggregation(node_features_list)
        elif aggregation_type == 'std':
            return self.std_aggregation(node_features_list)
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    def mean_aggregation(self, node_features_list):
        return torch.stack(node_features_list).mean(dim=0)

    def max_aggregation(self, node_features_list):
        return torch.stack(node_features_list).max(dim=0)[0]

    def min_aggregation(self, node_features_list):
        return torch.stack(node_features_list).min(dim=0)[0]

    def std_aggregation(self, node_features_list):
        return torch.stack(node_features_list).std(dim=0)


class RankAggregation(torch.nn.Module):
    def __init__(
        self,
        aggr_func: Literal["mean", "sum", "max", "std"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "relu",
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.update_func = update_func

    def update(self, inputs):
        if self.update_func == "sigmoid":
            return torch.sigmoid(inputs)
        if self.update_func == "relu":
            return torch.nn.functional.leaky_relu(inputs)
        if self.update_func == "tanh":
            return torch.tanh(inputs)
        return None

    def forward(self, x):
        if self.aggr_func == "sum":
            x = torch.sum(torch.stack(x), axis=0)
        if self.aggr_func == "mean":
            x = torch.mean(torch.stack(x), axis=0)
        if self.aggr_func == "max":
            x = torch.max(torch.stack(x), axis=0)[0] 
        if self.aggr_func == "std":
            x = torch.std(torch.stack(x), axis=0)

        if self.update_func is not None:
            x = self.update(x)
        return x