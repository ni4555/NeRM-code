import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Initialize a matrix with zeros of the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    # A simple heuristic could be based on the inverse of the demand normalized by the total demand
    # This will give higher weights to edges with higher demands
    heuristics_matrix = -distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    return heuristics_matrix