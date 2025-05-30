import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential value for each edge
    # Negative values for undesirable edges, positive for promising ones
    potential_value = -distance_matrix * normalized_demands

    # Return the potential value matrix
    return potential_value