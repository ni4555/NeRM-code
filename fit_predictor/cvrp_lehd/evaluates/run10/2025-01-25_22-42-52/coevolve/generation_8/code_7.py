import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of capacity each customer requires
    normalized_demands = demands / total_capacity
    
    # Compute the potential cost for each edge as the negative of the normalized demand
    # The negative sign is used to make the positive values more desirable
    potential_costs = -normalized_demands
    
    # Add a small constant to avoid division by zero when taking the log
    epsilon = 1e-6
    potential_costs = potential_costs + epsilon
    
    # Compute the logarithm of the potential costs to give a heuristic value
    heuristics = torch.log(potential_costs)
    
    # Subtract the minimum heuristic value from all to ensure all are positive
    min_heuristic = heuristics.min()
    heuristics = heuristics - min_heuristic
    
    return heuristics