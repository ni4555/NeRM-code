import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the normalized demand vector
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Create a vector of ones for the distance matrix
    ones = torch.ones_like(distance_matrix)

    # Compute the negative distance matrix to indicate undesirable edges
    negative_distance_matrix = -distance_matrix

    # Use a simple heuristic: sum of normalized demand times distance
    heuristics = (negative_distance_matrix + ones) * normalized_demands

    return heuristics