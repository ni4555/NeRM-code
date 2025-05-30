import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Compute the negative of the demands for use in the heuristic
    negative_demands = -normalized_demands

    # The heuristic can be a combination of negative demands and distance
    # Here, we are using the formula: heuristic = -demand + distance
    # This encourages the inclusion of edges with lower demand and lower distance
    heuristics = negative_demands + distance_matrix

    return heuristics