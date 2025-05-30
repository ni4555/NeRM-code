import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand
    total_demand = demands.sum()
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    # Calculate the heuristics based on the normalized demands
    heuristics = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    return heuristics