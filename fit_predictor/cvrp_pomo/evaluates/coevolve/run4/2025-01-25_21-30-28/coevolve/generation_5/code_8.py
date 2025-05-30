import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the demands to give higher priority to smaller demands
    inverse_demands = 1 / (normalized_demands + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate the heuristic value as the sum of the inverse demands and the negative distance
    # Negative distance to encourage shorter paths
    heuristics = inverse_demands - distance_matrix

    # Normalize the heuristics matrix to ensure the sum of heuristics for each row is equal to 1
    row_sums = heuristics.sum(dim=1, keepdim=True)
    heuristics = heuristics / row_sums

    return heuristics