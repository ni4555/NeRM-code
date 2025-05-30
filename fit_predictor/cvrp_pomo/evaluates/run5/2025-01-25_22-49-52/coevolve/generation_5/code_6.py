import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity

    # Calculate path potential based on distance and demand
    distance_potential = distance_matrix * demand_normalized

    # Normalize for consistent scaling
    max_potential = distance_potential.max()
    min_potential = distance_potential.min()
    normalized_potential = (distance_potential - min_potential) / (max_potential - min_potential)

    # Apply a penalty for high demand to encourage load balancing
    load_balance_penalty = demand_normalized * 0.1

    # Combine the normalized potential with the load balance penalty
    heuristics = normalized_potential - load_balance_penalty

    # Apply a small positive value to all edges to avoid zero values
    heuristics = torch.clamp(heuristics, min=0.001)

    return heuristics