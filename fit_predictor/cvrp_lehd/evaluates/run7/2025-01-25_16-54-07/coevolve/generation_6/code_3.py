import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate cumulative demand for each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    # Normalize cumulative demand by total vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    # Calculate the heuristic values based on normalized demand
    heuristics = normalized_demand * distance_matrix
    return heuristics