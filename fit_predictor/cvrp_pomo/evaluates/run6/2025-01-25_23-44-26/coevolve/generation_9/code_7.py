import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the distance matrix
    inverse_distance_matrix = 1 / distance_matrix

    # Compute the demand-sensitive penalty matrix
    penalty_matrix = normalized_demands[:, None] * normalized_demands[None, :] * inverse_distance_matrix

    # Apply the Inverse Distance Heuristic (IDH) by subtracting the penalty matrix from the inverse distance matrix
    heuristics_matrix = inverse_distance_matrix - penalty_matrix

    # Apply a smoothing technique to ensure uniformity across the problem instance
    heuristics_matrix = torch.clamp(heuristics_matrix, min=-1, max=1)

    return heuristics_matrix