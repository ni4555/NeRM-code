import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristic based on normalized demands
    initial_heuristic = -normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)

    # Adjust heuristic values based on edge weights
    adjusted_heuristic = initial_heuristic - distance_matrix

    # Normalize the heuristic values to ensure they are within a specific range
    max_heuristic = adjusted_heuristic.max()
    min_heuristic = adjusted_heuristic.min()
    normalized_heuristic = (adjusted_heuristic - min_heuristic) / (max_heuristic - min_heuristic)

    return normalized_heuristic