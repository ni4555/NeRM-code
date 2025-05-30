import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands between any two nodes
    demand_diff = demands[:, None] - demands[None, :]
    
    # Normalize the demand difference by the vehicle capacity
    demand_diff_normalized = demand_diff / demands.max()
    
    # Calculate the heuristic value as the negative of the normalized demand difference
    # This heuristic assumes that smaller demand differences are more promising
    heuristic_values = -demand_diff_normalized
    
    # Ensure that the heuristic values are within the specified range
    # We can set a lower bound to ensure no negative heuristic values are returned
    heuristic_values = torch.clamp(heuristic_values, min=0)
    
    return heuristic_values