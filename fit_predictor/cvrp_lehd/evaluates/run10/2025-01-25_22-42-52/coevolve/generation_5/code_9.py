import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential cost of visiting each customer
    potential_costs = -distance_matrix * normalized_demands

    # Add a small constant to avoid zero division
    epsilon = 1e-8
    potential_costs = potential_costs + epsilon

    # Normalize the potential costs to be between 0 and 1
    max_cost = potential_costs.max()
    min_cost = potential_costs.min()
    normalized_potential_costs = (potential_costs - min_cost) / (max_cost - min_cost)

    return normalized_potential_costs