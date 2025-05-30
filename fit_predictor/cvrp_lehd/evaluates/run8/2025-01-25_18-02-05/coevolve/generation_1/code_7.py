import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the heuristic values
    # We will use a simple heuristic where the heuristic value is inversely proportional to the demand
    # and inversely proportional to the distance from the depot (for edges connecting the depot)
    heuristics = -normalized_demands * distance_matrix

    # For edges that do not connect to the depot, we set the heuristic to a large negative value
    # to indicate they are not promising
    heuristics[distance_matrix == 0] = -torch.inf

    return heuristics