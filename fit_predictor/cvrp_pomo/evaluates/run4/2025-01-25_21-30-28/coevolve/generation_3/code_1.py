import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands to represent the relative load of each customer
    normalized_demands = demands / total_capacity
    
    # Calculate the average edge weight
    average_weight = (distance_matrix * demands).sum() / (demands.sum() ** 2)
    
    # Adjust heuristic values based on edge weight and demand
    # Negative values for undesirable edges, positive for promising ones
    heuristics = distance_matrix - average_weight * demands
    
    # Normalize heuristics to ensure load balancing
    heuristics = heuristics / total_capacity
    
    return heuristics