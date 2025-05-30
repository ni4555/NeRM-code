import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between nodes
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Normalize the demand difference to be between -1 and 1
    demand_diff = demand_diff / (demands.max() - demands.min())
    
    # Calculate the negative distance matrix (undesirable edges)
    negative_distance_matrix = -distance_matrix
    
    # Combine demand difference and negative distance matrix
    heuristics = negative_distance_matrix + demand_diff
    
    # Set the diagonal to zero to avoid considering self-loops
    torch.fill_diagonal_(heuristics, 0)
    
    return heuristics