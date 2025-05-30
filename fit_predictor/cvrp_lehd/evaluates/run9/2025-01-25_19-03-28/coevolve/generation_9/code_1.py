import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a matrix of negative distances
    negative_distance_matrix = -distance_matrix
    
    # Add normalized demands to the negative distance matrix
    heuristics_matrix = negative_distance_matrix + normalized_demands
    
    # Apply dynamic load balancing by prioritizing edges with lower load
    # We do this by multiplying each row by the demand of the corresponding node
    heuristics_matrix = heuristics_matrix * demands.unsqueeze(1)
    
    # Apply proximity-based route planning by prioritizing edges with shorter distance
    # This is already incorporated in the negative_distance_matrix
    
    return heuristics_matrix