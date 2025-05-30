import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (sum of demands)
    total_capacity = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_capacity
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Apply the greedy approach to assign routes based on demand and distance
    for i in range(1, len(normalized_demands)):
        for j in range(1, len(normalized_demands)):
            if j != i:
                # Calculate the heuristic value for the edge (i, j)
                heuristic_value = -normalized_distance_matrix[i, j] + normalized_demands[i]
                # Update the heuristic matrix
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix