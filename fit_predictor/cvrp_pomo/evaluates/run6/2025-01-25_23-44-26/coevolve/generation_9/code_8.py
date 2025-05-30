import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance heuristic
    idh_values = 1.0 / distance_matrix

    # Calculate the demand-sensitive penalty
    penalty = 1.0 / (1.0 + demands)

    # Combine the two heuristics
    combined_heuristics = idh_values * penalty

    # Normalize the combined heuristics to ensure uniformity
    max_heuristic = combined_heuristics.max()
    min_heuristic = combined_heuristics.min()
    normalized_heuristics = (combined_heuristics - min_heuristic) / (max_heuristic - min_heuristic)

    return normalized_heuristics