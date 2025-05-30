import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand factor for each edge based on the difference in demands
    demand_factor = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the distance-based heuristic
    distance_heuristic = -distance_matrix
    
    # Combine the demand factor and distance-based heuristic
    combined_heuristic = demand_factor * distance_heuristic
    
    # Normalize the combined heuristic to ensure it is within the desired range
    # This could be a simple min-max normalization or a more complex scaling
    min_combined = combined_heuristic.min()
    max_combined = combined_heuristic.max()
    normalized_heuristic = (combined_heuristic - min_combined) / (max_combined - min_combined)
    
    return normalized_heuristic