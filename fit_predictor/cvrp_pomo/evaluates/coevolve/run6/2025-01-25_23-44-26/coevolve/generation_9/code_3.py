import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to be between 0 and 1
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Calculate the inverse distance heuristic
    inv_distance = 1 / (distance_matrix + 1e-8)  # Add a small constant to avoid division by zero

    # Combine inverse distance with normalized demands
    heuristics = inv_distance * normalized_demands

    # Apply a demand-sensitive penalty mechanism to prevent overloading
    demand_penalty = demands / (demands.sum() + 1e-8)  # Avoid division by zero
    heuristics *= demand_penalty

    return heuristics