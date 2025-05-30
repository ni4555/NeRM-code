import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate inverse distance matrix
    inv_distance_matrix = 1.0 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics using inverse distance and demand normalization
    heuristics = -inv_distance_matrix * normalized_demands
    
    return heuristics