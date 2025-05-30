import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance matrix, which is the sum of the distance matrix
    # and the square of the demands. This will give a higher weight to edges that
    # have higher demands.
    total_distance_matrix = distance_matrix + demands ** 2

    # Normalize the total distance matrix by the maximum demand to ensure that
    # the weights are comparable across all edges.
    normalized_total_distance_matrix = total_distance_matrix / demands.max()

    # Generate a heuristics matrix by subtracting the normalized total distance
    # matrix from the distance matrix. Negative values indicate undesirable edges,
    # while positive values indicate promising ones.
    heuristics_matrix = distance_matrix - normalized_total_distance_matrix

    return heuristics_matrix