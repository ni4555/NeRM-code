import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate inverse distance heuristic
    inv_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate demand-penalty heuristic
    demand_penalty = -demands

    # Combine the inverse distance and demand-penalty heuristics
    combined_heuristic = inv_distance * normalized_demands + demand_penalty

    return combined_heuristic