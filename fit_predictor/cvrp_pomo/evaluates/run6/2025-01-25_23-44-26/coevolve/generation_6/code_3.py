import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic (IDH) values
    idh_values = 1 / distance_matrix

    # Apply demand penalty function
    demand_penalty = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    demand_penalty = torch.clamp(demand_penalty, min=0, max=1)  # Ensure non-negative values

    # Combine IDH and demand penalty to get initial heuristics
    heuristics = idh_values - demand_penalty

    return heuristics