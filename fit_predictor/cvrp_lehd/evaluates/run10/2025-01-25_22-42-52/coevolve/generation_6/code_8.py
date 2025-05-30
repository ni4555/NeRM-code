import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on normalized demands
    # The heuristic is a simple function of demand and distance
    # For example, we can use the negative of the demand as a heuristic for undesirable edges
    # and a positive value for promising edges (e.g., distance multiplied by demand)
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics