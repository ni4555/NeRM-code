import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()

    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand

    # Calculate the potential cost of each edge
    # The heuristic will be negative for edges that are not promising
    # and positive for edges that are promising
    heuristics = -distance_matrix * normalized_demands.expand_as(distance_matrix)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    heuristics = heuristics / (heuristics.abs() + epsilon)

    return heuristics