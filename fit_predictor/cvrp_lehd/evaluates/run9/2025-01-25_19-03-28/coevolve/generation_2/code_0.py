import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device and have the same dtype
    distance_matrix = distance_matrix.to(demands.device).to(torch.float32)
    demands = demands.to(distance_matrix.device).to(torch.float32)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential negative impact of each edge based on the demand
    demand_impact = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Calculate the potential positive impact of each edge based on the distance
    distance_impact = distance_matrix / distance_matrix.mean()
    
    # Combine the impacts to get the heuristic values
    heuristics = demand_impact + distance_impact
    
    # Ensure that the heuristic values are within the specified range (negative for undesirable, positive for promising)
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    
    return heuristics