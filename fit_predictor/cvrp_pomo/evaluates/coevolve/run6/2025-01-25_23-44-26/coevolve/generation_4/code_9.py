import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic: use reciprocal of distance as heuristic value
    # Assuming distance_matrix is a square matrix with non-negative values
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Add small constant to avoid division by zero

    # Demand penalty function: increase cost for edges leading to high demand on vehicles
    demand_penalty = normalized_demands[distance_matrix != 0] * 1000  # Example penalty factor

    # Combine inverse distance and demand penalty
    heuristic_values = inverse_distance - demand_penalty

    return heuristic_values