import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Calculate the normalized demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics using the formula:
    # heuristics = -distance * demand
    # This will give negative values for undesirable edges and positive values for promising ones
    heuristics = -distance_matrix * normalized_demands
    
    return heuristics