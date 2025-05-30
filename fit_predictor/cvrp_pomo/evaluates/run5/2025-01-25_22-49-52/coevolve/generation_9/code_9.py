import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential cost of each edge as the product of distance and normalized demand
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1)

    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    # Calculate the heuristic values as the ratio of potential costs to the sum of potential costs
    # This will give negative values for edges with lower potential costs and positive values for higher ones
    heuristic_values = potential_costs / (potential_costs.sum(1, keepdim=True) + epsilon)

    return heuristic_values