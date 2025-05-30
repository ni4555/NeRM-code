import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Compute the heuristic values as the product of inverse distance and normalized demand
    heuristic_values = inv_distance_matrix * normalized_demands
    
    return heuristic_values