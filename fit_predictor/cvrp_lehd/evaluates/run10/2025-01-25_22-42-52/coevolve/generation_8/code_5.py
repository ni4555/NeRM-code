import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative of the distance matrix for the heuristic calculation
    negative_distance_matrix = -distance_matrix

    # Compute the heuristic as a linear combination of the negative distances and demands
    heuristics = negative_distance_matrix * normalized_demands

    return heuristics