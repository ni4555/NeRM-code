import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands to get the fraction of each customer's demand
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost for each edge
    # We use a simple heuristic where the potential cost is the product of the distance and the normalized demand
    potential_costs = distance_matrix * normalized_demands.unsqueeze(1)
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    potential_costs = potential_costs / (potential_costs + epsilon)
    
    # Calculate the heuristic values by taking the log of the potential costs
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = torch.log(potential_costs)
    
    return heuristics