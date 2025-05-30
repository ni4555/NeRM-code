import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate inverse distance
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize demands by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Combine inverse distance and demand normalization
    heuristics = inv_distance * normalized_demands
    
    # Adjust the heuristics to have negative values for undesirable edges
    heuristics = 1.0 - heuristics
    
    return heuristics