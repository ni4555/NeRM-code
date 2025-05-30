import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    demands_sum = demands.sum()
    normalized_demands = demands / demands_sum

    # Create a tensor of the same shape as the distance matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j and i != 0:  # Skip the depot and self-edges
                heuristics[i, j] = distance_matrix[i, j] * normalized_demands[i] * (1 - demands[j])

    return heuristics