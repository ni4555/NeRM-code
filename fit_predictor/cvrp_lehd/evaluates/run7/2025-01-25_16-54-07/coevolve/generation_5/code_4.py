import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand to normalize
    total_demand = demands.sum()
    # Normalize the demands
    normalized_demands = demands / total_demand
    # Calculate the heuristics based on distance and demand
    heuristics = distance_matrix - normalized_demands.unsqueeze(1) * distance_matrix
    return heuristics