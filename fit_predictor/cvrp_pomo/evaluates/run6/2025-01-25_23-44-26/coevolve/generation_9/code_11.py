import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    vehicle_capacity = demands.max()
    normalized_demands = demands / vehicle_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the inverse distance heuristic
    heuristics = -distance_matrix

    # Incorporate demand-sensitive penalty mechanism
    # We add a penalty for edges that are close to the vehicle capacity limit
    demand_penalty = 0.1 * (1 - normalized_demands)
    heuristics += demand_penalty

    return heuristics