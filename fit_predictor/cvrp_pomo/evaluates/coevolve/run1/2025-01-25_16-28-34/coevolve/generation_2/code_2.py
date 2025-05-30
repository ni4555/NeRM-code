import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands (each customer demand divided by the total capacity)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the demands as a potential heuristic (higher demand = higher cost)
    # This encourages routes to be planned to fulfill the higher demand customers first
    inverse_demands = 1 / (normalized_demands + 1e-10)  # Adding a small value to avoid division by zero

    # The heuristic value is the product of distance and inverse demand, 
    # where a shorter distance to a higher demand customer is more promising
    # We use negative values to encourage the algorithm to avoid these edges
    heuristics = -distance_matrix * inverse_demands

    return heuristics