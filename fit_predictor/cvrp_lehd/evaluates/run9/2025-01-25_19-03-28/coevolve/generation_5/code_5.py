import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize demands to the range [0, 1]
    normalized_demands = demands / total_capacity

    # Compute the sum of normalized demands for each edge
    demand_sum = (normalized_demands * distance_matrix).sum(1)

    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Compute the difference between the demand sum and the maximum demand
    demand_diff = demand_sum - torch.max(demand_sum)

    # Use a simple heuristic: favor edges with lower difference
    heuristics = -demand_diff

    return heuristics