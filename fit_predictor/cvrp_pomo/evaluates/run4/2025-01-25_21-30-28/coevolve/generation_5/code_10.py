import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_normalized = demands / demands.sum()
    # Inverse distance heuristic
    inv_distance = 1.0 / (distance_matrix + 1e-10)  # Add a small value to avoid division by zero
    # Normalize the inverse distance by the demand to balance the influence of distance and demand
    normalized_inv_distance = inv_distance / (demands_normalized + 1e-10)
    return normalized_inv_distance