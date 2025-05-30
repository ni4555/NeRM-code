import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the distance matrix
    inv_distance_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate the demand-driven heuristic values
    demand_heuristic = inv_distance_matrix * normalized_demands

    # Integrate a penalty function for capacity constraints
    # Here we use a simple linear penalty function, but this can be replaced with a more complex one
    penalty_factor = 0.1
    capacity_penalty = torch.clamp(1 - demands, min=0) * penalty_factor

    # Combine the demand-driven heuristic with the capacity penalty
    combined_heuristic = demand_heuristic - capacity_penalty

    return combined_heuristic