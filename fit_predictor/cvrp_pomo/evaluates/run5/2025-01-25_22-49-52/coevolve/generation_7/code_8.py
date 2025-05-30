import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the potential value for each edge
    # Here we use a simple heuristic: the potential value is the negative of the distance
    # and we add a positive term proportional to the normalized demand
    potential_value = -distance_matrix + normalized_demands * 10.0

    # The heuristic function returns the potential value for each edge
    return potential_value