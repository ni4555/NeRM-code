import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the maximum demand a vehicle can carry
    max_demand = demands.max()
    
    # Calculate the potential savings for each edge
    potential_savings = (distance_matrix * demands).sum(dim=1) - (distance_matrix * max_demand).sum(dim=1)
    
    # Normalize the potential savings by the total demand
    normalized_savings = potential_savings / total_demand
    
    # Calculate the heuristic values
    heuristics = normalized_savings - torch.abs(normalized_savings)
    
    return heuristics