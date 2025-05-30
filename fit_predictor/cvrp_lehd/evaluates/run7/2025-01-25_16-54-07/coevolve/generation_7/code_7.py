import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the cumulative demand
    cumulative_demand = demands.cumsum(dim=0)
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    # Calculate the heuristic values based on normalized demand and distance
    heuristics = -distance_matrix * normalized_demand
    return heuristics