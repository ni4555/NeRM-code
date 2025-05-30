import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity

    # Calculate the heuristic values
    # For simplicity, we use a basic heuristic that is a combination of demand and distance
    # This is a placeholder for the actual heuristic that would be implemented
    heuristics = -normalized_demands * distance_matrix

    return heuristics