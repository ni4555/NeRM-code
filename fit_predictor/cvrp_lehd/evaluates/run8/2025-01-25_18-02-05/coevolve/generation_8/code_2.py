import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance heuristic (the closer the better)
    negative_distance_heuristic = -distance_matrix
    
    # Normalize the negative distance heuristic by the customer demands
    # to account for demand in the vicinity of the customers
    demand_normalized_heuristic = negative_distance_heuristic / demands
    
    # Subtract the demands to create a negative demand heuristic (the lower the better)
    negative_demand_heuristic = -demand_normalized_heuristic
    
    # Add the two heuristics to create a combined heuristic
    combined_heuristic = negative_distance_heuristic + negative_demand_heuristic
    
    return combined_heuristic