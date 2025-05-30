import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix for the heuristic component
    negative_distance = -distance_matrix

    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize the customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Compute the heuristic for each edge based on the sum of the negative distance and the normalized demand
    heuristics = negative_distance + normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    return heuristics