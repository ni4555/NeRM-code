import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the heuristics based on a simple formula:
    # Heuristic = -distance + demand
    # Negative distance indicates that the edge is undesirable, which we want to avoid
    # Positive demand indicates that the edge is promising, which we want to include
    heuristics = -distance_matrix + normalized_demands

    return heuristics