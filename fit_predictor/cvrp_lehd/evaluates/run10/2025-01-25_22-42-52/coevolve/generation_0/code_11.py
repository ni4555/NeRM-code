import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix to encourage shorter paths
    negative_distance_matrix = -distance_matrix

    # Calculate the normalized demand matrix where each customer demand is compared to the total vehicle capacity
    normalized_demands = demands / demands.sum()

    # Calculate the heuristics by multiplying the negative distances with the normalized demands
    heuristics = negative_distance_matrix * normalized_demands

    # Replace any negative values with zeros to indicate undesirable edges
    heuristics[heuristics < 0] = 0

    return heuristics