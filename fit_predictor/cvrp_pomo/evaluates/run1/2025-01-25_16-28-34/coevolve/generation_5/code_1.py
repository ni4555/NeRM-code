import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it's a single value for simplicity)
    total_capacity = demands.sum()

    # Normalize demands to represent the fraction of the total capacity each customer requires
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values
    # The heuristic function can be a simple inverse of the demand, as higher demand indicates a more urgent need
    # which could be considered as a higher priority for the route.
    # We also add a small constant to avoid division by zero
    heuristic_values = (1 / (normalized_demands + 1e-10)) * distance_matrix

    # Negative values for undesirable edges and positive values for promising ones
    # Here we use the negative of the heuristic values to ensure this condition
    heuristic_matrix = -heuristic_values

    return heuristic_matrix