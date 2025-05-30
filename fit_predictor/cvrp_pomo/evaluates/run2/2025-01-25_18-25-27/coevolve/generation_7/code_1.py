import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic for each edge
    # The heuristic is based on the difference in normalized demand and distance
    # This encourages edges that lead to underutilized vehicles and short distances
    heuristics = (normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)) - distance_matrix

    # Enforce capacity constraints by penalizing heavily overcapacity edges
    # This is done by subtracting the distance from a large number for overcapacity edges
    overcapacity_penalty = (demands.unsqueeze(1) > 1).float() * (n * n) * distance_matrix

    # Apply the penalty to the heuristics matrix
    heuristics = heuristics - overcapacity_penalty

    return heuristics