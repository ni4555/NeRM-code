import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as a product of normalized demands and distance
    # Negative values are for undesirable edges, positive for promising ones
    heuristics = -normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    return heuristics