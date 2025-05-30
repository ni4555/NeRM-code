import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance to ensure the heuristics are normalized
    max_distance = torch.max(distance_matrix)
    
    # Calculate the demand penalty for each edge based on customer demand
    demand_penalty = demands.unsqueeze(1) + demands.unsqueeze(0)
    demand_penalty = (demand_penalty * 2 - torch.sum(demand_penalty, dim=2)).unsqueeze(2)  # Subtract the sum of the demand from the demand of each edge
    
    # Calculate the distance-based penalty for each edge
    distance_penalty = -distance_matrix
    
    # Combine the demand and distance penalties
    heuristics = demand_penalty + distance_penalty
    
    # Normalize the heuristics to ensure they have positive values
    heuristics = heuristics / (max_distance + 1e-6)  # Add a small constant to avoid division by zero
    
    return heuristics