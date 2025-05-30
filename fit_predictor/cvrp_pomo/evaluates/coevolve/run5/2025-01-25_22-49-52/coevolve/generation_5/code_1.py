import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be in the range [0, 1]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the path potential based on distance and demand
    # The heuristic is negative for edges with higher distance or demand
    heuristic_matrix = -torch.abs(distance_matrix) - normalized_demands

    return heuristic_matrix