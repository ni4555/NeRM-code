import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the normalized demands
    demand_sum = demands.sum()
    
    # Compute the potential value for each edge as the negative of the distance
    # multiplied by the demand ratio
    potential = -distance_matrix * (demands / demand_sum)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential = potential + epsilon
    
    # Normalize the potential values to get a heuristic for each edge
    heuristic = potential / potential.sum()
    
    return heuristic