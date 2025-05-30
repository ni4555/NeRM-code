import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to have a sum of 1 for each vehicle
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the heuristics
    # Promising edges will have higher values, undesirable edges will have lower values
    heuristics = (distance_matrix - torch.abs(distance_matrix - distance_matrix.mean(dim=0))) * normalized_demands

    # Make sure that all values are in the range of negative to positive infinity
    heuristics = heuristics - heuristics.min()

    return heuristics