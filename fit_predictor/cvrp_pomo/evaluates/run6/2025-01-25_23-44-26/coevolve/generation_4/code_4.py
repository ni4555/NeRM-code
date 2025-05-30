import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the inverse distance for each edge, which is positive
    inverse_distances = 1.0 / distance_matrix

    # Calculate the demand penalty function, which is negative for high demands on full vehicles
    demand_penalty = (distance_matrix * normalized_demands).clamp(min=0.0)

    # Combine the inverse distance and demand penalty to get the heuristic values
    heuristics = inverse_distances - demand_penalty

    return heuristics