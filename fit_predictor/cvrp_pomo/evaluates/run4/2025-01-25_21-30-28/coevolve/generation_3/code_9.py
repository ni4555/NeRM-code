import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate heuristic values based on edge weights and demand
    # The heuristic is designed to be positive for promising edges and negative for undesirable ones
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix

    return heuristics