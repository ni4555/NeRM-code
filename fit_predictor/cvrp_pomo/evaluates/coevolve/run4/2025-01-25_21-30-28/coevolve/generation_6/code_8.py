import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_distance_matrix = distance_matrix / total_capacity

    # Use inverse distance as a heuristic
    inverse_distance_matrix = 1 / (normalized_distance_matrix + 1e-10)  # Add a small constant to avoid division by zero

    # Incorporate customer demands as a heuristic (load balancing)
    demand_heuristic = demands / total_capacity

    # Combine heuristics
    combined_heuristic = inverse_distance_matrix - demand_heuristic

    return combined_heuristic