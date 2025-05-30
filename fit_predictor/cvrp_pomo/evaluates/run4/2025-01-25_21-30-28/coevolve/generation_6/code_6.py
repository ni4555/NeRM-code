import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inverse_distance = 1.0 / distance_matrix

    # Combine the inverse distance with the normalized demands
    combined_heuristics = inverse_distance * normalized_demands

    # Normalize the combined heuristics to ensure they are all positive
    max_heuristic = combined_heuristics.max()
    heuristics = combined_heuristics - max_heuristic

    return heuristics