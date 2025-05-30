import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand difference
    demand_diff = demands - demands.mean()
    
    # Calculate the distance to the depot
    distance_to_depot = distance_matrix[:, 0]
    
    # Combine the normalized demand difference and the distance to the depot
    combined = demand_diff * distance_to_depot
    
    # Use the absolute value to ensure non-negative heuristics
    heuristics = torch.abs(combined)
    
    return heuristics