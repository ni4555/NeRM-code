import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize demands
    demands_normalized = demands / demands.sum()
    # Calculate the heuristics based on distance and demand
    # Using negative for undesirable edges, here assuming we have an undesirable heuristic base of 10
    base = torch.full_like(demands_normalized, 10)
    heuristics = -base + distance_matrix + demands_normalized.unsqueeze(1) * base.unsqueeze(0)
    return heuristics