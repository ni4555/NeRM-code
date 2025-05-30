import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential benefit of each edge
    # The benefit is defined as the product of the distance and the normalized demand
    # which is a common approach in many heuristics for the CVRP
    benefits = distance_matrix * normalized_demands

    # To ensure that the heuristic is meaningful, we can add a small constant
    # to avoid division by zero or very small values
    epsilon = 1e-8
    benefits = benefits + epsilon

    # The heuristic should return negative values for undesirable edges
    # and positive values for promising ones, so we take the negative of the benefits
    heuristics = -benefits

    return heuristics