import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with high negative values for undesirable edges
    heuristic_matrix = -torch.ones_like(distance_matrix)

    # Compute the potential benefit of each edge by considering the normalized demand
    # The benefit is inversely proportional to the demand (lower demand is better)
    benefit_matrix = 1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero

    # Use the benefit matrix to update the heuristic matrix
    # The heuristic is positive for promising edges and negative for undesirable ones
    heuristic_matrix = distance_matrix * benefit_matrix

    return heuristic_matrix