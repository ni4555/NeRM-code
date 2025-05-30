import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate heuristic for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate heuristic value based on distance and demand
                heuristic_value = distance_matrix[i, j] - normalized_demands[i] - normalized_demands[j]
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix