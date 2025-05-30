import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize customer demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristic (IDH) values
    idh_values = 1 / distance_matrix

    # Incorporate demand penalty function
    demand_penalty = (1 + 0.1 * (normalized_demands - 1))  # Adjust the penalty factor as needed

    # Combine IDH and demand penalty to get the heuristic values
    heuristic_values = idh_values * demand_penalty

    # Add a small constant to avoid division by zero
    heuristic_values = heuristic_values + 1e-6

    return heuristic_values