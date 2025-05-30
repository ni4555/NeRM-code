import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize a matrix of zeros with the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Calculate the negative distance for each edge and add the demand of the destination node
    # We use a small constant to avoid division by zero
    small_constant = 1e-8
    heuristics_matrix = (distance_matrix + demands.unsqueeze(1)).div(total_demand + small_constant)
    
    return heuristics_matrix