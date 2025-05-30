import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the distances
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristics using the inverse distance and normalized demands
    heuristics = inv_distance_matrix * normalized_demands
    
    return heuristics