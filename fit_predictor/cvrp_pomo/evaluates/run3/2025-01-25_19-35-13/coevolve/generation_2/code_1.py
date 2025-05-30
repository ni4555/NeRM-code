import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristics
    # For this example, we use a simple heuristic where we subtract the demand from the distance
    # to get a negative value for promising edges. The smaller the distance, the more promising the edge.
    heuristics = distance_matrix - (normalized_demands * distance_matrix)

    # Clip the values to ensure no negative heuristics
    heuristics = torch.clamp(heuristics, min=0)

    return heuristics