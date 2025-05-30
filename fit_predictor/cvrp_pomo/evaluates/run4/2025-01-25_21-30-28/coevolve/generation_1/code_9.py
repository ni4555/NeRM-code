import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load for each edge (i, j)
    load = demands * distance_matrix
    
    # Normalize the load by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_load = load / total_capacity
    
    # Calculate the heuristic values
    # We use a simple heuristic where edges with higher normalized load are considered less promising
    heuristics = -normalized_load
    
    return heuristics