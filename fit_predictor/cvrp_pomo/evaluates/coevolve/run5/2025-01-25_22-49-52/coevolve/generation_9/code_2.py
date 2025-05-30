import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands with respect to the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize a tensor with the same shape as distance_matrix, filled with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cost of each edge by adding the distance and a function of the demand
    # Here, we use a simple linear function: demand * demand_factor
    demand_factor = 1.0  # This factor can be adjusted to prioritize demand
    heuristics += distance_matrix * demand_factor * normalized_demands.unsqueeze(1)
    heuristics += normalized_demands.unsqueeze(0) * demand_factor

    # Enforce vehicle capacity constraint by penalizing high-demand edges
    # This is a simple heuristic, in a real problem, more complex rules might be applied
    heuristics[torch.where(demands > 1.0)] -= 1000

    return heuristics