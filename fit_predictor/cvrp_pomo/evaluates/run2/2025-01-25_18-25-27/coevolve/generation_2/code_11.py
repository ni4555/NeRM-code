import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands = demands / demands.sum()  # Normalize demands by the total vehicle capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic for each edge based on the ratio of demand to distance
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i, j] = demands[i] / distance_matrix[i, j]

    # Apply a penalty for edges that exceed the vehicle capacity
    for i in range(n):
        for j in range(n):
            if i != j:
                if demands[i] + demands[j] > 1:
                    heuristic_matrix[i, j] = -float('inf')

    return heuristic_matrix