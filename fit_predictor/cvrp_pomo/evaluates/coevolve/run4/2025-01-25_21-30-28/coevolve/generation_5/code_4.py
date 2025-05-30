import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Normalization heuristic
    normalization = normalized_demands

    # Combine heuristics: sum of inverse distance and normalization
    heuristics = inverse_distance + normalization

    return heuristics