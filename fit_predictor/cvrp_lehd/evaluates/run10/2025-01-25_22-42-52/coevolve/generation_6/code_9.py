import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a tensor with the same shape as distance_matrix filled with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_capacity
    
    # Calculate the negative heuristic for each edge (undesirable edges)
    # This is a simple heuristic where we consider the demand as a negative factor
    heuristics = -normalized_demands[torch.arange(n), None] - normalized_demands[None, :]
    
    # Calculate the positive heuristic for each edge (promising edges)
    # This can be a more complex heuristic, but for simplicity, let's use the negative of the distance matrix
    heuristics += distance_matrix
    
    return heuristics