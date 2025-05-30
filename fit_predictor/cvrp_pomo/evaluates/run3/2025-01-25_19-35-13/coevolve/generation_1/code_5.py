import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along each edge
    cumulative_demand = torch.cumsum(demands, dim=0) - demands

    # Normalize the cumulative demand by the vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()

    # Calculate the heuristic values as the negative of the normalized demand
    heuristics = -normalized_demand

    return heuristics