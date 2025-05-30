import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on normalized demands
    # We use a simple heuristic where we consider edges with lower demands and shorter distances
    # as more promising. This is a basic approach and can be replaced with more sophisticated
    # heuristics depending on the problem specifics.
    heuristic_values = -normalized_demands * distance_matrix
    
    # We can add more sophisticated heuristics here, such as:
    # - Incorporating load balance (e.g., penalizing heavily loaded vehicles)
    # - Considering the proximity to the depot (e.g., giving priority to closer customers)
    
    return heuristic_values