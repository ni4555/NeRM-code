import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics based on the normalized demands
    # For this simple heuristic, we use the negative of the normalized demand as the heuristic value
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -normalized_demands
    
    return heuristics