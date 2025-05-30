import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to handle varying scales of node distances
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Normalize the demands to the total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Calculate the potential function based on distance and demand
    potential_function = normalized_distance_matrix * normalized_demands

    # Employ an epsilon value to prevent division by zero errors
    epsilon = 1e-8
    potential_function = torch.clamp(potential_function, min=-epsilon)

    # Evaluate the desirability of routes using the potential function
    # Negative values for undesirable edges, positive values for promising ones
    heuristics = -potential_function

    return heuristics