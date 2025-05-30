import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each row (each node)
    row_sums = normalized_demands.sum(dim=1, keepdim=True)
    
    # Calculate the heuristics based on the normalized demands
    # We use a simple heuristic where the more demand a node has, the more promising it is
    # to include it in the route. This is a basic example and can be replaced with more
    # sophisticated heuristics.
    heuristics = -row_sums
    
    return heuristics