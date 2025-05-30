import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential cost of visiting each customer
    potential_costs = distance_matrix * normalized_demands
    
    # Add a term that penalizes visiting customers with high demand
    demand_penalty = (1 - normalized_demands) * 100  # Example penalty, can be adjusted
    potential_costs += demand_penalty
    
    # The heuristic function should have positive values for promising edges
    # and negative values for undesirable edges
    return potential_costs