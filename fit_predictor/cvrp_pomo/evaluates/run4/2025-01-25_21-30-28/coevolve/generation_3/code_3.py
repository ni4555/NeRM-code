import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic for each edge based on demand
    heuristic_matrix[:, 1:] = distance_matrix[:, 1:] * (1 - normalized_demands[1:])

    # Add a penalty for edges leading to the depot from non-depot nodes
    penalty = -torch.inf
    heuristic_matrix[1:, 0] = penalty
    heuristic_matrix[0, 1:] = penalty

    return heuristic_matrix