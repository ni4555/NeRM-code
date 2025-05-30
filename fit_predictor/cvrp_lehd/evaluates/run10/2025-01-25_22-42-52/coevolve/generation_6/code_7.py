import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the cost matrix based on distance and demand
    cost_matrix = distance_matrix * normalized_demands

    # Create a penalty for high demand edges
    penalty = 100 * (cost_matrix > 0).float()

    # Add a negative heuristic value for high demand edges
    heuristics = -cost_matrix + penalty

    return heuristics