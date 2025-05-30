import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector by the sum of all demands
    demands_sum = torch.sum(demands)
    normalized_demands = demands / demands_sum

    # Create a tensor with all ones for the edge costs
    edge_costs = torch.ones_like(distance_matrix)

    # Calculate the relative demands for each customer
    relative_demands = demands / demands_sum

    # Compute the heuristic value for each edge
    heuristics = (1 - normalized_demands) * (1 - relative_demands) * distance_matrix

    # Ensure the heuristic values are within the range of negative infinity to 1
    heuristics = torch.clamp(heuristics, min=float('-inf'), max=1.0)

    return heuristics