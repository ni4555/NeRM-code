import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand normalized by the total vehicle capacity
    total_demand = demands.sum()
    
    # Calculate the potential negative heuristic for each edge
    # This is a simplistic approach that assumes higher demand edges are more promising
    negative_heuristic = -demands / total_demand
    
    # Calculate the potential positive heuristic for each edge
    # Here we use the distance as a proxy for the cost of the edge
    positive_heuristic = distance_matrix
    
    # Combine the negative and positive heuristics
    # This is a simple linear combination, but other methods could be used
    combined_heuristic = negative_heuristic + positive_heuristic
    
    # Ensure that all negative values are set to a very low negative value
    # to indicate undesirable edges
    combined_heuristic[combined_heuristic < 0] = -torch.inf
    
    return combined_heuristic