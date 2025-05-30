import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be in the range [0, 1]
    normalized_demands = demands / demands.sum()
    
    # Compute inverse distance matrix
    inverse_distance_matrix = 1 / distance_matrix
    
    # Compute weighted demands for each edge based on the demand of the destination
    # and the inverse distance
    weighted_demands = normalized_demands.unsqueeze(1) * inverse_distance_matrix
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    weighted_demands = weighted_demands + epsilon
    
    # Normalize the weighted demands matrix so that the sum of values in each row is 1
    heuristic_matrix = weighted_demands / weighted_demands.sum(dim=1, keepdim=True)
    
    return heuristic_matrix