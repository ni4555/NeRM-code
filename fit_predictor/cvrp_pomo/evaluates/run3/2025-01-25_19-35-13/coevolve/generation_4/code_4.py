import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()

    # Calculate the heuristic values using the nearest neighbor heuristic
    # This heuristic assigns a positive value to edges that are part of the shortest path
    # from the depot to each customer and a negative value to the edges that are not.
    # For the edges to the depot, we assign the negative of the total demand.
    heuristics = -normalized_demands

    # The distance_matrix is used to compute the actual distances and assign positive values
    # to the shortest paths. For simplicity, we'll just use the distance to the nearest neighbor
    # as a proxy for the shortest path.
    heuristics[distance_matrix == 0] = 0  # Set the diagonal to 0
    heuristics[distance_matrix != 0] = distance_matrix[distance_matrix != 0].min(dim=1).values

    return heuristics