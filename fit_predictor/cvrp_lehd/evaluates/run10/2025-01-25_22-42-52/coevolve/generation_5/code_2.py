import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized
    demands = demands / demands.sum()

    # Calculate Manhattan distance for each edge, this will be used as the heuristic value
    # The Manhattan distance heuristic is simply the sum of the absolute differences
    # in the coordinates for each dimension.
    heuristic_matrix = torch.abs(distance_matrix - demands.unsqueeze(1)).sum(dim=2)

    # We use a simple heuristic where higher distances are penalized, so we negate the values
    # for a more intuitive positive heuristic where lower values are better.
    heuristic_matrix = -heuristic_matrix

    return heuristic_matrix