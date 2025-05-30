import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize distance matrix by dividing by the maximum distance
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Compute potential value for each edge
    potential_value = normalized_distance_matrix * normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    potential_value = torch.clamp(potential_value, min=epsilon)
    
    # Invert the potential value to have negative values for undesirable edges
    heuristics = -potential_value
    
    return heuristics