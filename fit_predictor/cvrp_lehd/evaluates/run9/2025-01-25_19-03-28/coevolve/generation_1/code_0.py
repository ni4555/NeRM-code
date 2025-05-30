import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the demand potential for each edge
    demand_potential = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Apply a penalty to edges that exceed the vehicle capacity
    capacity_penalty = (demands.unsqueeze(0) > 1.0).float() * -1e5
    
    # Calculate the heuristics score by summing the demand potential and the capacity penalty
    heuristics = demand_potential + capacity_penalty
    
    return heuristics