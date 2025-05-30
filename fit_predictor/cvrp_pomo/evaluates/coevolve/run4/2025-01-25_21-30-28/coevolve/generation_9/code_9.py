import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    inverse_distance = 1.0 / distance_matrix

    # Normalize customer demands
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Combine the inverse distance heuristic with the demand normalization
    combined_heuristic = inverse_distance * normalized_demands

    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    combined_heuristic = combined_heuristic + epsilon

    # Invert the heuristic values to have negative values for undesirable edges
    heuristics = -combined_heuristic

    return heuristics