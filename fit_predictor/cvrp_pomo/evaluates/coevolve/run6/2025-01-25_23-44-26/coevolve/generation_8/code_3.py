import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix

    # Incorporate load balancing into the heuristic
    load_balance = demands.unsqueeze(1) * demands.unsqueeze(0)
    load_balance = (load_balance / total_capacity).clamp(min=0)

    # Combine the inverse distance and load balance into the heuristic
    heuristic_matrix = inverse_distance - load_balance

    return heuristic_matrix