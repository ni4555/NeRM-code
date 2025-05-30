import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse distance matrix
    inverse_distance_matrix = 1 / distance_matrix
    
    # Normalize the inverse distance matrix by the sum of normalized demands
    normalized_inverse_distance_matrix = inverse_distance_matrix / normalized_demands.unsqueeze(1)
    
    # Apply a simple heuristic by summing the normalized inverse distances
    heuristics_matrix = normalized_inverse_distance_matrix.sum(dim=1)
    
    # Return the heuristics matrix, which has the same shape as the distance matrix
    return heuristics_matrix