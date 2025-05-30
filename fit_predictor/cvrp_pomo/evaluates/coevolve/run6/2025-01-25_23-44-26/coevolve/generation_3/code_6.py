import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize the distance matrix to ensure all values are between 0 and 1
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    # Calculate the heuristic values based on normalized demands and distances
    # We use the formula: heuristic = demand * distance
    # Negative values for edges with high demand and/or high distance
    heuristics = -normalized_demands * normalized_distance_matrix

    return heuristics