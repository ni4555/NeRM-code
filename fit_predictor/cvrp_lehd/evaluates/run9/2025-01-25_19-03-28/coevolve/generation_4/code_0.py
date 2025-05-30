import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to represent the fraction of the total capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the demand heuristic
    demand_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the distance heuristic
    distance_heuristic = -distance_matrix
    
    # Combine demand and distance heuristics
    combined_heuristic = demand_heuristic + distance_heuristic
    
    # Clip negative values to zero to ensure the heuristic is non-negative
    combined_heuristic = torch.clamp(combined_heuristic, min=0)
    
    return combined_heuristic