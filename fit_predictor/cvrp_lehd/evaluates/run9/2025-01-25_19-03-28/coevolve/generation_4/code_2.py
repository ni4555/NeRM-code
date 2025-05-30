import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential cost of visiting each customer
    # This is a simple heuristic based on demand, but it can be replaced with more complex heuristics
    potential_costs = -normalized_demands
    
    # Subtract the distance matrix from the potential costs to get the heuristics
    heuristics = potential_costs - distance_matrix
    
    return heuristics