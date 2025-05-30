import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inv_distance = 1 / distance_matrix

    # Implement demand penalty function
    demand_penalty = normalized_demands * demands
    demand_penalty[distance_matrix == 0] = 0  # Avoid division by zero

    # Combine inverse distance and demand penalty
    heuristics = inv_distance - demand_penalty

    return heuristics