import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Initialize heuristics with zero values
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate heuristic values for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Heuristic based on distance and normalized demand
                heuristics[i, j] = distance_matrix[i, j] - demands_normalized[i] * demands_normalized[j]

    return heuristics