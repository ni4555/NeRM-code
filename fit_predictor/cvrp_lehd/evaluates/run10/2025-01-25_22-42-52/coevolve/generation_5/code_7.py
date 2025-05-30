import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of all customer demands
    total_capacity = demands.sum()
    
    # Normalize the customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the negative heuristic values based on customer demands
    # Negative values for undesirable edges
    negative_heuristics = -normalized_demands
    
    # Calculate the positive heuristic values based on the distance matrix
    # Positive values for promising edges
    # We use the distance matrix directly here as a simple example, but
    # more sophisticated distance-based heuristics could be implemented
    positive_heuristics = distance_matrix
    
    # Combine the negative and positive heuristics
    combined_heuristics = negative_heuristics + positive_heuristics
    
    return combined_heuristics