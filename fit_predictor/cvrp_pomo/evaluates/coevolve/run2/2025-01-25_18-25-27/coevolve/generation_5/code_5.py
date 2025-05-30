import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize a heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Compute the heuristics for each edge
    # For each edge (i, j), calculate the heuristic as:
    # - distance(i, j) * (demands[j] / total_capacity)
    # This heuristic is negative for undesirable edges and positive for promising ones
    heuristics_matrix = distance_matrix * normalized_demands.unsqueeze(0)

    return heuristics_matrix