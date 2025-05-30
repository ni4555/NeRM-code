import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative of the normalized demand as a heuristic
    # Negative values indicate undesirable edges (high demand)
    heuristics = -normalized_demands

    return heuristics