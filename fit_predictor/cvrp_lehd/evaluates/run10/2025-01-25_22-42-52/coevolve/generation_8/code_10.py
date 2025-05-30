import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to sum to 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Calculate the potential of each edge
    # The potential is the product of the inverse of the distance and the normalized demand
    potential_matrix = torch.inverse(distance_matrix) * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_matrix = potential_matrix + epsilon

    # Calculate the heuristics values
    # Negative values for undesirable edges (high distance or zero demand)
    # Positive values for promising edges
    heuristics = -torch.log(potential_matrix)

    return heuristics