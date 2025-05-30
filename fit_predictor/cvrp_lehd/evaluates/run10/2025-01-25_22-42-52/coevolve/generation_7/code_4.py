import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge based on the normalized demand
    # and the distance between nodes
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_costs = potential_costs + epsilon
    
    # Calculate the heuristic values as the negative of the potential costs
    heuristics = -potential_costs
    
    return heuristics