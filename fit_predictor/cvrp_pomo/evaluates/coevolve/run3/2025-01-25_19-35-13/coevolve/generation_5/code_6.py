import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values
    # For this simple heuristic, we can use the normalized demand as a measure of promise
    # This is a basic approach and might not be the most efficient or effective for CVRP
    heuristics = normalized_demands * distance_matrix
    
    return heuristics