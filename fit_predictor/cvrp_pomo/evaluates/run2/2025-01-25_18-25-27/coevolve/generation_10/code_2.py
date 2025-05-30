import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristics
    # The heuristic is a combination of the normalized demand and the distance to the depot
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -distance_matrix[:, 0] + normalized_demands

    return heuristics