import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_vector = demands / total_capacity

    # Normalize the distance matrix to account for demands
    normalized_distance_matrix = distance_matrix * demand_vector.unsqueeze(1) * demand_vector.unsqueeze(0)

    # Implementing a simple heuristic: the lower the normalized distance, the more promising the edge
    # For simplicity, we can use the negative of the normalized distance as the heuristic value
    heuristics = -normalized_distance_matrix

    return heuristics