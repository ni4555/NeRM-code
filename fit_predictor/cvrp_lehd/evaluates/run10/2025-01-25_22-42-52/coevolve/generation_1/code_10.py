import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized
    total_capacity = demands[0]
    normalized_demands = demands[1:] / total_capacity

    # Compute the negative of the distance matrix
    negative_distance_matrix = -distance_matrix

    # Add the negative demands to the negative distance matrix
    # The demand at the depot is ignored as it's not relevant for the heuristics
    heuristics = negative_distance_matrix + normalized_demands

    return heuristics