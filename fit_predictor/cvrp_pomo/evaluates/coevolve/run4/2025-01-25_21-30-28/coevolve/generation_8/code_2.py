import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Inverse distance heuristic
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero
    
    # Combine inverse distance and demand normalization
    combined_heuristic = inverse_distance * normalized_demands
    
    # Apply a scaling factor to adjust the heuristics
    scaling_factor = 10  # This can be tuned for better performance
    heuristics = combined_heuristic * scaling_factor
    
    return heuristics