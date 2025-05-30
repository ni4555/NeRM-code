import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of each customer's demand
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge
    # This is a simple heuristic that assumes the cost is proportional to the demand
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_costs = potential_costs + epsilon
    
    # Calculate the heuristic values
    # We use a simple heuristic where we subtract the potential cost from 1
    # This gives us a value between 0 and 1, where 1 indicates a promising edge
    heuristics = 1 - potential_costs
    
    return heuristics