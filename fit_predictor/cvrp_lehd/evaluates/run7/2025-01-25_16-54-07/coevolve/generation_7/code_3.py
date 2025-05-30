import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cumulative demand mask
    cumulative_demand_mask = demands.cumsum(0)
    
    # Normalize cumulative demand by the total vehicle capacity
    normalized_cumulative_demand = cumulative_demand_mask / demands.sum()
    
    # Compute the heuristic values
    heuristics = -distance_matrix + normalized_cumulative_demand
    
    # Adjust the heuristic values to ensure they are negative for undesirable edges
    heuristics[heuristics >= 0] = -heuristics[heuristics >= 0] + 1e-10
    
    return heuristics