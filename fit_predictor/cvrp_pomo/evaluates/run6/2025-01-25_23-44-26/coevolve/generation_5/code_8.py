import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()  # Assuming vehicle capacity is equal to total demand

    # Demand penalty function: higher demand customers closer to capacity get higher penalties
    demand_penalty = 1 + demands / vehicle_capacity

    # Normalize distance matrix by demand penalty to adjust for customer demand
    adjusted_distance = distance_matrix / demand_penalty

    # Inverse distance heuristic: edges with lower adjusted distance are more promising
    # We use negative values to indicate undesirable edges (for minimization)
    heuristics = -adjusted_distance

    return heuristics