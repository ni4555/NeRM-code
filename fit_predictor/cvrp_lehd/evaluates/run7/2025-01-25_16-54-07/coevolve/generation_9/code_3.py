import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node considering the depot
    cumulative_demand = demands + demands[0]
    
    # Normalize the cumulative demand by the total vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    
    # Calculate the heuristic values based on normalized demand and distance
    # Negative values for undesirable edges (high demand or high distance)
    # Positive values for promising edges (low demand or low distance)
    heuristic_values = -normalized_demand * distance_matrix
    
    return heuristic_values