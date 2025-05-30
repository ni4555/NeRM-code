import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    max_distance = distance_matrix.max()
    min_distance = distance_matrix.min()
    normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the sum of demands along the diagonal
    diagonal_demand_sum = demands.sum()
    
    # Create a tensor of all ones for the same shape as the demands vector
    ones = torch.ones_like(demands)
    
    # Calculate the heuristic value for each edge
    heuristic_matrix = -torch.abs(normalized_distance_matrix - demands.unsqueeze(0) - demands.unsqueeze(1)) - diagonal_demand_sum * ones
    
    return heuristic_matrix