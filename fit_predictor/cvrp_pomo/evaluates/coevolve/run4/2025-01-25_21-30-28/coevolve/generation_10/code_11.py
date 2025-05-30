import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic values
    inverse_distance = 1 / distance_matrix
    
    # Normalize the demands to get the Normalization heuristic values
    normalized_demands = demands / demands.sum()
    
    # Calculate the sum of heuristics
    combined_heuristics = inverse_distance + normalized_demands
    
    # Subtract the sum from the maximum value to ensure all values are negative for undesirable edges
    max_combined_heuristic = combined_heuristics.max()
    heuristics = combined_heuristics - max_combined_heuristic
    
    return heuristics