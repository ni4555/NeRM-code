import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value for each edge as a negative of the demand
    # This will give negative values for promising edges (since we want to minimize distance)
    potential = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Add distance to the potential to make longer distances less promising
    heuristics = potential + distance_matrix

    return heuristics