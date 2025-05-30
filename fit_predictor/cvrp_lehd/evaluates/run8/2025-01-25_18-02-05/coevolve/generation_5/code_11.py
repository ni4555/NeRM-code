import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand from each customer to the depot
    demand_diff = demands - demands[0]

    # Compute the heuristic values based on demand difference and distance
    # Negative values are for undesirable edges, positive for promising ones
    heuristics = -demand_diff * distance_matrix

    return heuristics