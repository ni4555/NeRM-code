import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each row (customer)
    row_sums = normalized_demands.sum(dim=1, keepdim=True)
    
    # Calculate the heuristics based on the distance matrix and normalized demands
    # We use the following heuristic: heuristics[i, j] = -distance[i, j] * (1 - demand[i] / row_sum[i])
    # This will give negative values for edges that are less promising, and positive values for more promising ones
    heuristics = -distance_matrix * (1 - normalized_demands / row_sums)
    
    return heuristics