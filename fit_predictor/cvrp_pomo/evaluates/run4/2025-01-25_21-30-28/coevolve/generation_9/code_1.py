import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inverse_distance = 1 / distance_matrix

    # Calculate the demand normalization heuristic
    normalized_demands = demands / demands.sum()

    # Combine the heuristics using element-wise multiplication
    combined_heuristics = inverse_distance * normalized_demands

    # Ensure the heuristic values are negative for undesirable edges and positive for promising ones
    # by subtracting the maximum possible value of combined_heuristics from all elements
    max_heuristic = combined_heuristics.max()
    heuristics = combined_heuristics - max_heuristic

    return heuristics