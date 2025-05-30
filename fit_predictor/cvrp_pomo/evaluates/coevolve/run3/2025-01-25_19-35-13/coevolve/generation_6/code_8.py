import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics based on the normalized demands
    # We use a simple heuristic where the higher the demand, the more promising the edge
    heuristics = normalized_demands * distance_matrix
    
    return heuristics