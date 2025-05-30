import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Inverse distance heuristic: prioritize nearby nodes
    inverse_distance = 1.0 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero

    # Demand normalization heuristic: balance the allocation of customer demands
    demand_heuristic = normalized_demands

    # Combine the heuristics
    combined_heuristic = inverse_distance + demand_heuristic

    # Ensure that the heuristic matrix is symmetric (since distance matrix is symmetric)
    combined_heuristic = (combined_heuristic + combined_heuristic.t()) / 2

    return combined_heuristic