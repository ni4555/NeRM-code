import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate inverse distance
    inv_distance = 1.0 / distance_matrix

    # Apply demand penalty for high demand customers close to capacity
    demand_penalty = normalized_demands * demands
    demand_penalty[distance_matrix == 0] = 0  # Avoid division by zero for depot node

    # Combine inverse distance and demand penalty
    heuristics = inv_distance - demand_penalty

    return heuristics