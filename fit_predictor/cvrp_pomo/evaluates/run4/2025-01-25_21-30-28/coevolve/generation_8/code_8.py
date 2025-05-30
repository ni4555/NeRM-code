import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Inverse distance heuristic
    inverse_distance = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Demand normalization heuristic
    normalized_demands = demands / demands.sum()

    # Calculate the heuristic value for each edge
    # We use a simple weighted sum of inverse distance and normalized demand
    heuristic_matrix = inverse_distance * normalized_demands

    return heuristic_matrix