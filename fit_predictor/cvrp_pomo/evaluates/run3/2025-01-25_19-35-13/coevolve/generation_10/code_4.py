import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of normalized demands for each row (customer)
    row_sums = normalized_demands.sum(dim=1, keepdim=True)
    
    # Calculate the heuristics as the negative sum of normalized demands for each edge
    heuristics = - (row_sums * distance_matrix).sum(dim=1)
    
    return heuristics