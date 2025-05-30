import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristics based on normalized demands
    initial_heuristics = normalized_demands * total_capacity

    # Apply a penalty for edges that connect to the depot to avoid starting/ending at the depot
    depot_penalty = torch.full_like(initial_heuristics, -1e9)
    initial_heuristics[distance_matrix == 0] = depot_penalty[distance_matrix == 0]

    # Apply a penalty for edges that lead to overloading
    # Assuming that the maximum capacity is a predefined constant
    max_capacity = 1.0
    overloading_penalty = torch.where(
        initial_heuristics < -max_capacity,
        torch.full_like(initial_heuristics, -1e9),
        torch.zeros_like(initial_heuristics)
    )
    initial_heuristics += overloading_penalty

    return initial_heuristics