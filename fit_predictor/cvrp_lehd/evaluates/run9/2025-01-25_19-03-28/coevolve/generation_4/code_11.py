import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity by summing the normalized demands
    vehicle_capacity = demands.sum()
    
    # Calculate the negative heuristic value based on distance and demand
    # Negative heuristic values are undesirable edges
    negative_heuristic = -distance_matrix
    
    # Calculate a positive heuristic value that considers demand
    # The idea is to encourage visiting customers with higher demands
    # We normalize this by the vehicle capacity to ensure it's scale-invariant
    positive_heuristic = demands / vehicle_capacity * distance_matrix
    
    # Combine the negative and positive heuristics
    heuristics = negative_heuristic + positive_heuristic
    
    # We could also add more complexity, such as:
    # - Penalizing edges that would cause the vehicle to exceed capacity
    # - Rewarding edges that lead to a more balanced vehicle load
    # - Incorporating a time factor for service response times
    # For simplicity, we will not implement these additional complexities here

    return heuristics