import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized
    total_capacity = demands.sum()
    demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristics for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            # The heuristic value is calculated as the negative demand multiplied by the distance
            # The idea is to prioritize shorter distances and higher demands
            heuristic_value = -demands[i] * distance_matrix[i][j]
            # Set the heuristic value in the matrix
            heuristic_matrix[i][j] = heuristic_value

    return heuristic_matrix