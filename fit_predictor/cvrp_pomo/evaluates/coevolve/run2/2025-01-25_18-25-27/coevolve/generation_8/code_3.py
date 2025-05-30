import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demands_normalized = demands / total_capacity

    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the heuristic for each edge based on distance and demand
    heuristics += distance_matrix
    heuristics -= demands_normalized

    # Apply real-time penalties to prevent overloading
    penalties = torch.abs(demands_normalized) * 0.1
    heuristics += penalties

    return heuristics