import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    # Initialize a tensor with the same shape as the distance matrix, filled with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the demand-to-capacity ratio for each customer
    demand_to_capacity_ratio = demands / demands.sum()
    
    # Calculate the initial heuristic values based on demand-to-capacity ratio
    heuristic_matrix[depot_index, 1:] = -demand_to_capacity_ratio[1:]
    heuristic_matrix[1:, depot_index] = -demand_to_capacity_ratio[1:]
    
    # Add distance penalties for edges
    heuristic_matrix += distance_matrix
    
    # Normalize the heuristic matrix to ensure non-negative values
    min_value = heuristic_matrix.min()
    heuristic_matrix -= min_value
    
    return heuristic_matrix