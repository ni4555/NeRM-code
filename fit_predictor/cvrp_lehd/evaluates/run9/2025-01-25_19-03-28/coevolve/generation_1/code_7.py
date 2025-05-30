import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the total distance matrix
    total_distance = distance_matrix.sum(dim=1)

    # Compute a heuristic based on normalized demands and total distances
    # For simplicity, we'll use the product of normalized demand and total distance
    # This heuristic assumes that more demanded and distant edges are less promising
    heuristics = -normalized_demands * total_distance

    return heuristics