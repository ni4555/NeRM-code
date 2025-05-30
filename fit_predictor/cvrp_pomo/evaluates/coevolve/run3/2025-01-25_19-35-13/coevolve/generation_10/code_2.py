import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    # Calculate the heuristic values based on normalized demands
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i == j:
                # No distance to the depot itself
                heuristic_matrix[i, j] = -float('inf')
            else:
                # Calculate the heuristic value based on normalized demand
                heuristic_matrix[i, j] = -normalized_demands[i] * distance_matrix[i, j]
    return heuristic_matrix