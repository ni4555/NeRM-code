import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum demand among all customers
    max_demand = torch.max(demands)
    
    # Normalize the demands by the maximum demand
    normalized_demands = demands / max_demand
    
    # Calculate the load for each edge
    load = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the heuristic value for each edge
    heuristic_values = -load
    
    # Add a small constant to avoid zeros to ensure the heuristic is differentiable
    epsilon = 1e-8
    heuristic_values = heuristic_values + epsilon
    
    return heuristic_values