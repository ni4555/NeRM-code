import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    epsilon = 1e-10
    total_demand = demands.sum()
    demand_vector = demands / (total_demand + epsilon)  # Normalize demands
    distance_vector = distance_matrix / (distance_matrix.sum(axis=1) + epsilon)  # Normalize distances

    # Calculate the combined heuristics
    combined_heuristics = demand_vector * distance_vector

    # Apply a penalty for edges that connect the same node
    self_loop_penalty = torch.eye(n, dtype=combined_heuristics.dtype) * -float('inf')
    combined_heuristics = combined_heuristics + self_loop_penalty

    return combined_heuristics