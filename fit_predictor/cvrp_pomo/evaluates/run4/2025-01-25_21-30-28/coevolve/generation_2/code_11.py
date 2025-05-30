import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand
    total_demand = demands.sum()
    # Normalize demands to the capacity
    normalized_demands = demands / total_demand
    # Calculate the heuristics based on demand and distance
    heuristics = normalized_demands * distance_matrix
    return heuristics