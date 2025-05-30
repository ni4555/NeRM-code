import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand mask
    cumulative_demand = demands.cumsum(dim=0)
    
    # Normalize cumulative demand to prevent overcapacity
    normalized_demand = cumulative_demand / demands.sum()
    
    # Compute the heuristic based on normalized demand and distance
    heuristic = -distance_matrix + normalized_demand
    
    return heuristic