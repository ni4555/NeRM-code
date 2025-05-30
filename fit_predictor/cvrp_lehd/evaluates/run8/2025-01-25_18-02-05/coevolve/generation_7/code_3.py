import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the inverse of the normalized demands to create a heuristic
    inverse_demands = 1 / normalized_demands
    
    # Use the distance matrix to compute the heuristic values
    # We use a simple heuristic that combines the inverse demand and distance
    # For example, a common heuristic is 1 / (distance + demand)
    heuristic_values = inverse_demands / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero
    
    return heuristic_values