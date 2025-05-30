import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the normalized demands to use as a heuristic
    inverse_normalized_demands = 1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero

    # Calculate the distance-based heuristic using the Inverse Distance Heuristic (IDH)
    # Subtracting the distance to penalize longer distances
    distance_based_heuristic = -distance_matrix

    # Combine the demand-based and distance-based heuristics
    combined_heuristic = inverse_normalized_demands * distance_based_heuristic

    # Apply a penalty function for capacity constraints
    # Increase the heuristic value for edges that would cause the vehicle to be over capacity
    # This is a simplified penalty function that assumes a linear relationship between
    # the heuristic value and the fraction of capacity used
    capacity_penalty = (1 - demands) * combined_heuristic

    # The final heuristic matrix
    final_heuristic = capacity_penalty

    return final_heuristic