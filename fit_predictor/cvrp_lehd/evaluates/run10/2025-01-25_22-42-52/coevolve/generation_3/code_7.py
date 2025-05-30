import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (sum of all demands)
    total_capacity = demands.sum()
    
    # Calculate the normalized demand vector
    normalized_demands = demands / total_capacity
    
    # Initialize a heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Vectorized implementation of the heuristic function
    heuristics_matrix = heuristics_matrix - (normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0))
    
    return heuristics_matrix