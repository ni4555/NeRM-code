import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the negative of the normalized demands to use as the heuristic
    # Negative values indicate undesirable edges (high demand)
    negative_heuristics = -normalized_demands
    
    # Calculate the sum of distances in the distance matrix
    # This will be used to normalize the distance-based heuristic
    distance_sum = distance_matrix.sum()
    
    # Normalize the distance matrix by the sum of distances
    # This will be used as the heuristic for edges with low demand
    normalized_distance_matrix = distance_matrix / distance_sum
    
    # Combine the demand-based and distance-based heuristics
    # We add the two heuristics to combine their information
    combined_heuristics = negative_heuristics + normalized_distance_matrix
    
    return combined_heuristics