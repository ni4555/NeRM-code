import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    depot_index = 0
    # Calculate the cost of visiting each customer from the depot
    cost_to_customers = distance_matrix[depot_index, 1:]
    # Normalize the cost by the customer demand
    normalized_cost = cost_to_customers / demands[1:]
    # Calculate the heuristics value: higher cost per unit demand is less promising
    heuristics = -normalized_cost
    return heuristics