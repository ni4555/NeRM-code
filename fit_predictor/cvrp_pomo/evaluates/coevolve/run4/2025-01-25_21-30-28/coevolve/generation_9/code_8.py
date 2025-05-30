import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic
    inverse_distance = 1.0 / distance_matrix

    # Combine the inverse distance heuristic with the demand normalization
    combined_heuristic = inverse_distance * normalized_demands

    # Adjust the heuristic values to be negative for undesirable edges and positive for promising ones
    # This is done by subtracting the maximum value to ensure that all values are negative for undesirable edges
    max_combined_heuristic = combined_heuristic.max()
    adjusted_heuristic = combined_heuristic - max_combined_heuristic

    return adjusted_heuristic