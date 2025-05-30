import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands vector
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values based on the normalized demands
    # Negative values for undesirable edges (high demand)
    # Positive values for promising edges (low demand)
    heuristics = -normalized_demands * distance_matrix
    
    return heuristics