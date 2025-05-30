import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity

    # Calculate the cost of visiting each customer from the depot
    cost_to_customer = distance_matrix[0, 1:]

    # Calculate the cost of returning to the depot from each customer
    cost_from_customer = distance_matrix[1:, 0]

    # Combine the costs to get the total cost for each edge
    total_cost = cost_to_customer + cost_from_customer

    # Calculate the heuristic value for each edge
    # The heuristic value is the total cost multiplied by the customer demand
    # and normalized by the total capacity
    heuristic_values = total_cost * demand_normalized

    # Invert the heuristic values to have negative values for undesirable edges
    # and positive values for promising ones
    heuristics = -heuristic_values

    return heuristics