import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix
    max_distance = torch.max(distance_matrix)
    
    # Normalize the distance matrix with respect to the maximum distance
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Compute the inverse of the normalized demands
    inverse_demands = 1 / (demands / demands.sum())
    
    # Compute the heuristic values as a product of the normalized distances and inverse demands
    heuristics = normalized_distance_matrix * inverse_demands
    
    # Subtract the maximum heuristic value to ensure all heuristics are negative for undesirable edges
    heuristics -= torch.max(heuristics)
    
    return heuristics