import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demand vector
    total_demand = demands.sum()
    
    # Normalize the demand vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values as the negative of the normalized demand
    # Negative values indicate undesirable edges
    heuristic_matrix = -normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Add the distance matrix to the heuristic matrix to encourage shorter paths
    heuristic_matrix += distance_matrix
    
    return heuristic_matrix