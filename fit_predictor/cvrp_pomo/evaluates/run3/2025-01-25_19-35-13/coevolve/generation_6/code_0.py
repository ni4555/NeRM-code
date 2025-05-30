import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands vector is normalized by the total vehicle capacity
    total_capacity = demands[0]
    demands = demands / total_capacity

    # Calculate the heuristics based on the inverse of the distance and normalized demand
    # We want to minimize the route distance, so we use negative distances and negative demands
    heuristics = -distance_matrix - demands

    # We need to ensure that no vehicle leaves the depot with zero capacity,
    # so we add a small constant to the diagonal to prevent zero values
    diagonal = torch.diag(heuristics)
    diagonal.add_(1e-5)
    heuristics = heuristics.masked_fill(torch.eq(diagonal, 0), 0)

    return heuristics