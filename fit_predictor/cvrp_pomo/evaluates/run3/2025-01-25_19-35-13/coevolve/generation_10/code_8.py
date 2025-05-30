import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the heuristic values based on normalized demands and distances
    # We will use a simple heuristic: the product of normalized demand and normalized distance
    heuristics = normalized_demands * normalized_distance_matrix
    
    # We can add a small constant to avoid zeros to ensure that the edges are not considered undesirable
    epsilon = 1e-6
    heuristics = heuristics + epsilon
    
    return heuristics