import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential heuristics based on normalized demands
    # The heuristic for an edge is the negative of the normalized demand from the source node
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix

    return heuristics