import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the potential cost of each edge
    # This is a simplistic heuristic that assumes the potential cost is the demand of the destination
    heuristics = normalized_demands

    # Apply a penalty for edges that exceed the vehicle capacity
    # Here we use a simple penalty of -1 for edges that would cause an overloaded vehicle
    # This is a placeholder for a more sophisticated constraint-aware allocation strategy
    for i in range(1, len(demands)):
        if heuristics[i] > 1:
            heuristics[i] = -1

    return heuristics