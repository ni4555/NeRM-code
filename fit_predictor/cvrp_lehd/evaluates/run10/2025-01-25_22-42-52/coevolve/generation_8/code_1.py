import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge as the negative of the demand
    # This heuristic assumes that edges with higher demands are more promising
    potential_costs = -normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_costs += epsilon
    
    # Calculate the heuristic value as the inverse of the potential cost
    # This encourages the algorithm to include edges with lower potential costs
    heuristic_values = 1 / potential_costs
    
    return heuristic_values