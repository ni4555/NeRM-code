import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the demand to capacity ratio for each customer
    demand_to_capacity_ratio = demands / demands.sum()  # Normalize by the total demand

    # Calculate the savings for each edge
    # Savings is defined as the demand of the customer + the cost of the trip
    # In this heuristic, we'll use the demand to capacity ratio to represent the cost of the trip
    savings = demand_to_capacity_ratio + torch.log(1.0 / (1.0 - demand_to_capacity_ratio))

    # Subtract savings from the distance to get a heuristic score
    # Higher scores correspond to edges that are more promising
    heuristic_matrix = distance_matrix - savings

    return heuristic_matrix