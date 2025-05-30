import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Compute the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.mean()

    # Calculate the potential value for each edge
    potential_values = normalized_distance_matrix - normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)

    return potential_values