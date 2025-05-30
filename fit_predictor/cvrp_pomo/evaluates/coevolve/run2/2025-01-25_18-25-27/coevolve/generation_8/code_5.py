import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize heuristic matrix with large negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # Calculate heuristic values for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate heuristic based on distance and normalized demand
                heuristic = distance_matrix[i][j] - normalized_demands[i] * normalized_demands[j]
                heuristic_matrix[i][j] = heuristic
    
    return heuristic_matrix