import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with the same shape as the distance matrix, filled with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the total vehicle capacity as the sum of all customer demands
    total_capacity = demands.sum()
    
    # Normalize the demands by the total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of distances from the depot to each customer
    sum_distances = torch.sum(distance_matrix, dim=1)
    
    # Compute the heuristics based on normalized demands and sum of distances
    # Heuristics are negative for undesirable edges and positive for promising ones
    heuristics = -distance_matrix + sum_distances * normalized_demands
    
    return heuristics