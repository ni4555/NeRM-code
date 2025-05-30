import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize customer demands
    normalized_demands = demands / demands.sum()
    # Create a vector of ones
    ones = torch.ones_like(normalized_demands)
    # Calculate the heuristics as the sum of demands and distance matrix
    # Subtracting the distance to penalize longer paths
    heuristics = normalized_demands + distance_matrix - ones * distance_matrix
    return heuristics