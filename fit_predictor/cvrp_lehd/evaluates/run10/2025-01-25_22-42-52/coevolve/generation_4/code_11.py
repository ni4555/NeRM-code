import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / total_capacity
    
    # Calculate the heuristic value as the negative of the normalized distance plus demand
    # Negative values for undesirable edges, positive values for promising ones
    heuristics = -normalized_distance_matrix + demands
    
    return heuristics