import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on normalized demands
    # The heuristic is a combination of the normalized demand and the distance to the depot
    heuristics = distance_matrix[:, 0] * normalized_demands

    # Adjust the heuristic values to be negative for undesirable edges and positive for promising ones
    # This is done by subtracting the maximum heuristic value from all heuristic values
    max_heuristic = heuristics.max()
    heuristics -= max_heuristic

    return heuristics