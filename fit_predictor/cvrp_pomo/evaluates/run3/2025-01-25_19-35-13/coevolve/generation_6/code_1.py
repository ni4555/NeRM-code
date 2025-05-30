import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_sum = demands.sum()
    if demands_sum == 0:
        return torch.zeros_like(distance_matrix)
    
    # Normalize the distance matrix to account for demand
    normalized_distance_matrix = distance_matrix / demands_sum
    
    # Add demand-based weights to the distance matrix
    heuristics_matrix = normalized_distance_matrix - demands
    
    # Avoid negative values by setting them to zero
    heuristics_matrix = torch.clamp(heuristics_matrix, min=0)
    
    return heuristics_matrix