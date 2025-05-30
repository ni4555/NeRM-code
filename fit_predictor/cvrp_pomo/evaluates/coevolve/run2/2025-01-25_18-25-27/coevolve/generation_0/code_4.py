import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    normalized_demands = demands / demands.sum()
    
    # Calculate the negative distance heuristic
    negative_distance_heuristic = -distance_matrix
    
    # Calculate the demand heuristic
    demand_heuristic = demands
    
    # Combine the two heuristics with a weighted sum
    # Here, we assume a weight of 0.5 for each heuristic, but this can be adjusted
    combined_heuristic = 0.5 * (negative_distance_heuristic + demand_heuristic)
    
    # Ensure the heuristic matrix does not contain NaNs or Infs
    combined_heuristic = torch.clamp(combined_heuristic, min=float('-inf'), max=float('inf'))
    
    return combined_heuristic