import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity (sum of all demands)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the "distance heuristic" which is the product of the distance and the normalized demand
    distance_heuristic = distance_matrix * normalized_demands

    # Subtract the maximum value for each row from the row sums to ensure non-negative values
    row_sums = distance_heuristic.sum(dim=1, keepdim=True)
    max_per_row = distance_heuristic.max(dim=1, keepdim=True)[0]
    adjusted_heuristic = distance_heuristic - max_per_row - row_sums

    # The resulting heuristic matrix has positive values for promising edges and negative values for undesirable ones
    return adjusted_heuristic