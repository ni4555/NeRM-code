import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic
    inverse_distances = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Demand penalty function
    demand_penalty = normalized_demands * (1 + 0.1 * (demands > 0.5).float())  # Increase cost for high demands

    # Combine inverse distance and demand penalty
    heuristics = -inverse_distances + demand_penalty

    return heuristics