import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of all customer demands
    total_capacity = demands.sum()
    
    # Normalize the customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the initial heuristics based on normalized demands
    heuristics = normalized_demands
    
    # Adjust heuristics based on distance matrix to prioritize closer customers
    heuristics = heuristics * distance_matrix
    
    # Apply a dampening factor to ensure non-negative heuristics
    dampening_factor = 0.5
    heuristics = heuristics * dampening_factor
    
    # Introduce a small random noise to avoid local optima
    noise = torch.rand_like(heuristics) * 0.01
    heuristics = heuristics - noise
    
    return heuristics