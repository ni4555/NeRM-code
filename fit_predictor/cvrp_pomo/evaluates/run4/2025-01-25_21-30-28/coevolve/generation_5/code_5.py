import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize distance matrix to create a weight matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Inverse distance heuristic
    inverse_distance = 1 / (normalized_distance_matrix + 1e-10)  # Adding small value to avoid division by zero
    
    # Demands heuristic: higher demands get lower priority
    demands_heuristic = -normalized_demands
    
    # Combine heuristics
    heuristics = inverse_distance + demands_heuristic
    
    return heuristics