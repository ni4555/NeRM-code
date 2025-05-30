import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Create a matrix to hold the heuristics
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristics for each edge
    # A positive heuristic value is assigned to edges with high demand relative to distance
    heuristics_matrix = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)

    return heuristics_matrix