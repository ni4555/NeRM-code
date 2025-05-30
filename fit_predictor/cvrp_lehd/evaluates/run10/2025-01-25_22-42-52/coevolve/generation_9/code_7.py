import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge
    potential_costs = distance_matrix * normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_costs = potential_costs / (potential_costs + epsilon)
    
    # Calculate the heuristics by subtracting the potential costs from 1
    heuristics = 1 - potential_costs
    
    return heuristics