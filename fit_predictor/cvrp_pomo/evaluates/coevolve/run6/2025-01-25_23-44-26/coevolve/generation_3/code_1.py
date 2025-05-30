import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values for each edge
    # Here we are using a simple heuristic where the weight is the negative of the distance
    # and a penalty for high demand
    heuristics = -distance_matrix + (normalized_demands * 1000)  # Example penalty for high demand

    return heuristics