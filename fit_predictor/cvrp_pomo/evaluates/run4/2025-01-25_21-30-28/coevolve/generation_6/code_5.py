import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the size of the distance matrix
    n = distance_matrix.size(0)
    
    # Calculate the inverse of the demands
    inverse_demands = 1 / demands
    
    # Calculate the normalization factor based on the total vehicle capacity
    normalization_factor = demands.sum() / demands
    
    # Normalize the inverse demands
    normalized_inverse_demands = inverse_demands / normalization_factor
    
    # Calculate the heuristics using the normalized inverse demands
    heuristics = normalized_inverse_demands * distance_matrix
    
    # Return the heuristics matrix
    return heuristics