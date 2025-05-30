import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance heuristic, which is negative for shorter distances
    negative_distance_heuristic = -distance_matrix
    
    # Calculate the demand-based heuristic, which is the demand of the customer node
    demand_heuristic = demands
    
    # Combine the two heuristics, giving more weight to the negative distance heuristic
    combined_heuristic = negative_distance_heuristic + demand_heuristic
    
    # Ensure that the values are within the desired range, e.g., [-1, 1]
    combined_heuristic = torch.clamp(combined_heuristic, min=-1, max=1)
    
    return combined_heuristic