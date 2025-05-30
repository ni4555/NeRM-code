import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to have a sum of 1
    demands_normalized = demands / demands.sum()
    
    # Compute the heuristic values
    # The heuristic is a function of both distance and demand
    # For simplicity, we can use a linear combination: distance * demand_weight
    # where demand_weight is a function of the demands to give higher priority to higher demand nodes
    demand_weight = demands_normalized / demands_normalized.sum()
    heuristic_values = distance_matrix * demand_weight
    
    return heuristic_values