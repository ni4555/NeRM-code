import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by total vehicle capacity
    demands_normalized = demands / total_demand
    
    # Calculate the inverse of the demand to use as a heuristic
    demand_inverse = 1 / demands_normalized
    
    # Use the inverse of the demand as a heuristic for the edges
    # This assumes that edges with lower demand are more promising
    heuristics = -demand_inverse
    
    # Add a small constant to avoid division by zero
    heuristics = heuristics + 1e-8
    
    return heuristics