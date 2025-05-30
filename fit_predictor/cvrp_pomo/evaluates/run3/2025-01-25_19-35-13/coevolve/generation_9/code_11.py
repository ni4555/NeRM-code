import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands to identify the most urgent demands
    demand_diff = (demands[1:] - demands[:-1]).abs()
    
    # Create a matrix with the same shape as the distance matrix initialized to -inf
    heuristics = torch.full_like(distance_matrix, fill_value=float('-inf'))
    
    # Set the diagonal to zero (no distance to itself)
    torch.fill_diagonal_(heuristics, 0)
    
    # Set the heuristics for the most urgent demands to 0 (immediate visit)
    heuristics[torch.arange(len(demands) - 1), torch.arange(1, len(demands))] = 0
    heuristics[torch.arange(1, len(demands)), torch.arange(len(demands) - 1)] = 0
    
    # Update heuristics based on the difference in demands
    heuristics[1:, 0] = 0
    heuristics[0, 1:] = 0
    heuristics[1:, 1:] = (distance_matrix[1:, 1:] + demand_diff)
    heuristics[0, 1:] = (distance_matrix[0, 1:] + demand_diff)
    
    # Normalize the heuristics matrix to ensure non-negative values
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics