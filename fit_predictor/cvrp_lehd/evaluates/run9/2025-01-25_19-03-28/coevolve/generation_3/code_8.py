import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are both 1D tensors with shape [n]
    n = distance_matrix.size(0)
    # Calculate the inverse of the demands
    inv_demands = 1.0 / (demands + 1e-10)  # Add a small constant to avoid division by zero
    # Compute the normalized distance matrix
    normalized_distance = distance_matrix / (distance_matrix.max() + 1e-10)  # Avoid division by zero
    # Combine the demand-based weights and distance-based weights
    heuristic_values = (normalized_distance * inv_demands).unsqueeze(1) * inv_demands.unsqueeze(0)
    # Add a large negative value for the depot to the diagonal (self-loop)
    diagonal_mask = torch.eye(n, device=distance_matrix.device)
    heuristic_values = heuristic_values - 1e10 * diagonal_mask
    return heuristic_values