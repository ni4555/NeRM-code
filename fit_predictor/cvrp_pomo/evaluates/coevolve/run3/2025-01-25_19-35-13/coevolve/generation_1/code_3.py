import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands vector
    total_demand = demands.sum()
    
    # Normalize the demands vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize a tensor of the same shape as the distance matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge
    # For each edge, the heuristic is the negative of the distance multiplied by the normalized demand
    # This encourages edges with smaller distances and higher demands to be more promising
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics