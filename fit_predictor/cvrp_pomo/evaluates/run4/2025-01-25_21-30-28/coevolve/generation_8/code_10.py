import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance matrix
    inv_distance_matrix = 1.0 / (distance_matrix + 1e-6)  # Add a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics based on inverse distance and demand normalization
    heuristics = inv_distance_matrix * normalized_demands
    
    return heuristics