import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to avoid overflow issues
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the cumulative sum of demands to determine the load at each node
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Compute a heuristic based on the load difference at each node
    load_diff = torch.abs(cumulative_demand - demands)
    
    # Use a simple heuristic that encourages paths that balance the load
    # We subtract the load difference from the distance to give a heuristic value
    heuristics = distance_matrix - load_diff
    
    # Add a small constant to ensure that all heuristics are positive
    heuristics = heuristics + 1e-6
    
    return heuristics