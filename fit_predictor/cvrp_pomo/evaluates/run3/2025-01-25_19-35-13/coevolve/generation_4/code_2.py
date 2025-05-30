import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the sum of all demands
    total_demand = demands.sum()
    # Normalize demands by total capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Compute the heuristic values
    for i in range(1, n):
        for j in range(1, n):
            if demands[j] > 0:  # Only consider non-depot nodes with non-zero demand
                heuristic_value = normalized_demands[j] * (distance_matrix[i, j] - distance_matrix[0, j])
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix