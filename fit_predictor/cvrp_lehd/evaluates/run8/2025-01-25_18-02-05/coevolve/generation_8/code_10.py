import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance heuristic
    neg_distance_heuristic = -distance_matrix
    
    # Calculate the demand heuristic
    demand_heuristic = demands
    
    # Combine the two heuristics
    combined_heuristic = neg_distance_heuristic + demand_heuristic
    
    return combined_heuristic