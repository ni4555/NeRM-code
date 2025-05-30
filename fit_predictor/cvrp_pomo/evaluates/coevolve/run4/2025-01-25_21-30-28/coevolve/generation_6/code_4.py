import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Combine the inverse distance heuristic with the normalized demands
    # The idea is to give higher priority to edges with lower distance and higher demand
    combined_heuristics = inverse_distance * normalized_demands
    
    # Apply normalization to ensure the values are within a reasonable range
    max_value = combined_heuristics.max()
    min_value = combined_heuristics.min()
    normalized_combined_heuristics = (combined_heuristics - min_value) / (max_value - min_value)
    
    return normalized_combined_heuristics