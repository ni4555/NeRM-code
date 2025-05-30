import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demand by total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic
    inv_distance_heuristic = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Load balancing heuristic
    load_balance_heuristic = (normalized_demands * (1 + 1 / distance_matrix)).T / (demands.T + 1e-8)

    # Combine heuristics
    combined_heuristic = inv_distance_heuristic * load_balance_heuristic

    # Apply a demand penalty function based on current load
    demand_penalty = 1 / (1 + demands)
    combined_heuristic *= demand_penalty

    return combined_heuristic