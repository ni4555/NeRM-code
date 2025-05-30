import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize a matrix to store heuristic values
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the inverse of the distance matrix (heuristic for IDH)
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero

    # Apply demand-driven weighting to the inverse distance heuristic
    weighted_heuristics = inverse_distance * normalized_demands

    # Define a penalty function for capacity constraints
    # For this example, we'll use a simple linear penalty
    capacity_penalty = (1 - demands / total_capacity) * 1000  # Higher penalty for edges that are over capacity

    # Combine the weighted heuristic with the capacity penalty
    heuristics_matrix = weighted_heuristics - capacity_penalty

    return heuristics_matrix