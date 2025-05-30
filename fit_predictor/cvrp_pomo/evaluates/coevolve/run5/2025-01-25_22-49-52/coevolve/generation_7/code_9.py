import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize distance matrix by dividing by the maximum distance
    normalized_distance = distance_matrix / distance_matrix.max()

    # Compute potential value for each edge
    # We use a simple heuristic: the negative of the normalized distance
    # multiplied by the normalized demand (to give a preference to edges with lower demand)
    potential = -normalized_distance * normalized_demands

    # To make the potential values more distinct, we add a small constant
    potential += 1e-5

    return potential