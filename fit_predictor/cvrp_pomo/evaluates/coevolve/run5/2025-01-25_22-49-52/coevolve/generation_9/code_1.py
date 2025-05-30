import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of demand as a heuristic for promising edges
    demand_inverse = 1 / (normalized_demands + 1e-8)  # Add a small constant to avoid division by zero

    # Initialize a matrix with zeros of the same shape as the distance matrix
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Vectorized implementation of the heuristic
    # Promising edges will have positive values, undesirable edges will have negative values
    # The heuristic is based on the inverse of demand
    heuristic_matrix = -distance_matrix * demand_inverse

    return heuristic_matrix