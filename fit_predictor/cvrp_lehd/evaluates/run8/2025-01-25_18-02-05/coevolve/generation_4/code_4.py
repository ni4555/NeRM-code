import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the difference between demands and 1 (assuming maximum capacity per vehicle is 1)
    demand_diff = 1 - normalized_demands

    # Calculate the heuristic values as the product of the demand difference and the distance
    # Negative values indicate undesirable edges (high demand), positive values indicate promising edges (low demand)
    heuristics = demand_diff * distance_matrix

    return heuristics