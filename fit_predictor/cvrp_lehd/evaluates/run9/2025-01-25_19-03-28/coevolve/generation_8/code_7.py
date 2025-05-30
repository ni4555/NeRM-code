import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on distance and normalized demand
    # We use a simple heuristic where the more distant the customer, the less promising the edge
    # and the higher the demand, the more promising the edge
    heuristics = distance_matrix * normalized_demands

    # Subtract the maximum heuristic value to ensure non-negative values
    max_heuristic = heuristics.max()
    heuristics -= max_heuristic

    return heuristics