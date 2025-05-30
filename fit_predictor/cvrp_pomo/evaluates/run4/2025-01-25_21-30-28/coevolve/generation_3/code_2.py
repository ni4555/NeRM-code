import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the initial heuristic values
    heuristics = -distance_matrix

    # Adjust heuristic values based on normalized demands
    heuristics += normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)

    # Ensure that the heuristics matrix has the same shape as the distance matrix
    heuristics = heuristics.view_as(distance_matrix)

    return heuristics