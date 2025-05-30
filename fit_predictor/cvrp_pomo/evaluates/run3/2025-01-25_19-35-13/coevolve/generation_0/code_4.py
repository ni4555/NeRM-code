import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demand vector does not include the depot demand
    demands = demands[1:]
    
    # Calculate the sum of demands and normalize by vehicle capacity
    total_demand = demands.sum()
    vehicle_capacity = 1.0  # Assuming the total vehicle capacity is 1 for normalization
    
    # Compute the heuristics: negative for high demand, positive for short distance
    heuristics = (1 - demands) * distance_matrix
    
    # Normalize the heuristics based on total demand and vehicle capacity
    heuristics /= total_demand / vehicle_capacity
    
    # Adjust signs to have negative values for undesirable edges
    heuristics = -heuristics
    
    return heuristics