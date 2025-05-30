import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the minimum distance from the depot to each customer
    min_distances = torch.min(distance_matrix[:, 1:], dim=0).values
    
    # Normalize the minimum distances by the total vehicle capacity
    normalized_distances = min_distances / total_demand
    
    # Calculate the heuristics: negative values for longer distances and higher demands
    heuristics = -normalized_distances - demands
    
    return heuristics