import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the cumulative sum of demands to use for heuristics
    cumulative_demands = torch.cumsum(demands, dim=0)
    # Calculate the heuristics based on the cumulative demands
    heuristics = -torch.abs(distance_matrix - cumulative_demands)
    return heuristics