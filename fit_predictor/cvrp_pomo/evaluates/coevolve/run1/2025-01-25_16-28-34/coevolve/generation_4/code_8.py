import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate normalized demand
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the distance-weighted demand
    distance_weighted_demand = torch.mul(distance_matrix, normalized_demands)

    # Initialize the heuristic matrix with negative values
    heuristic_matrix = -torch.ones_like(distance_matrix)

    # Apply the greedy approach to assign routes based on distance-weighted demand
    for i in range(1, distance_matrix.shape[0]):
        for j in range(1, distance_matrix.shape[0]):
            heuristic_matrix[i, j] = distance_weighted_demand[i, j]

    # Normalize the heuristic matrix to ensure positive values
    heuristic_matrix = (heuristic_matrix - heuristic_matrix.min()) / (heuristic_matrix.max() - heuristic_matrix.min())

    return heuristic_matrix