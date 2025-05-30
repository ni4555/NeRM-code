import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Normalize the difference by the vehicle capacity
    normalized_diff = demand_diff / demands.sum()
    
    # Compute the heuristic value based on distance and demand difference
    heuristic_matrix = -distance_matrix + normalized_diff
    
    return heuristic_matrix