import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the heuristic values for each edge
    # Here we use a simple heuristic that is a function of the normalized demand and distance
    # This is a placeholder for a more sophisticated heuristic that would be defined here
    heuristic_values = -normalized_demands * distance_matrix

    return heuristic_values