import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate cumulative demand along each potential route (excluding the depot)
    cumulative_demand = demands.cumsum(0)[1:]  # Exclude the depot (index 0)

    # Calculate normalized demand, which is the cumulative demand divided by the total capacity
    normalized_demand = cumulative_demand / demands[0]

    # Create a heuristics matrix initialized with negative infinity
    heuristics = -torch.inf * torch.ones_like(distance_matrix)

    # For each customer (excluding the depot), assign a positive heuristic value
    # that is proportional to the normalized demand and inversely proportional to the distance
    heuristics[1:, 1:] = -normalized_demand[None, :] * distance_matrix[1:, 1:]

    return heuristics