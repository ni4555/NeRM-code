import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity
    
    # Compute the heuristics by multiplying the distance by the normalized demand
    heuristics = distance_matrix * demands
    
    return heuristics