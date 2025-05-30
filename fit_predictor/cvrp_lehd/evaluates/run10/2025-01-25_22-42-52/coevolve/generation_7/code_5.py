import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand as a scalar
    total_demand = demands.sum()
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristics based on the inverse of distance (promising edges) and
    # a function of normalized demand (unpromising edges)
    heuristics = 1 / distance_matrix + normalized_demands
    
    # Ensure that the values are negative for undesirable edges and positive for promising ones
    # by adding a very small constant to avoid division by zero
    epsilon = 1e-6
    heuristics = torch.where(heuristics < 0, -epsilon, heuristics)
    
    return heuristics