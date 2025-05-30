import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values: distance multiplied by the normalized demand
    heuristics = distance_matrix * normalized_demands
    
    return heuristics