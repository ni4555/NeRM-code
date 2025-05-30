import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential function based on normalized demands
    potential_function = distance_matrix * normalized_demands

    # Apply a negative sign to undesirable edges
    # We can consider edges with zero demand or zero distance as undesirable
    undesirable_edges = (potential_function == 0) | (distance_matrix == 0)
    potential_function[undesirable_edges] *= -1

    return potential_function