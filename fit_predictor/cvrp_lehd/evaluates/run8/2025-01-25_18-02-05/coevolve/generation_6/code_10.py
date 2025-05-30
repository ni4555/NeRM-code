import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix for a heuristic based on distance
    heuristic_matrix = -distance_matrix

    # Normalize by the sum of demands to account for the vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    # Calculate a demand-based heuristic
    demand_heuristic = normalized_demands.repeat(distance_matrix.shape[0], 1)

    # Combine the distance heuristic with the demand heuristic
    combined_heuristic = heuristic_matrix + demand_heuristic

    # To ensure that the matrix contains negative values for undesirable edges,
    # we add a small positive value to the entire matrix
    combined_heuristic = combined_heuristic + 1e-6

    return combined_heuristic