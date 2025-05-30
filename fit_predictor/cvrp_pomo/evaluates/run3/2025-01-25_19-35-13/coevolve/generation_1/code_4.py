import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands by the total capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values as the negative of the normalized demand values
    # Negative values for undesirable edges, positive for promising ones
    heuristics = -normalized_demands
    
    # Convert the heuristics to a tensor with the same shape as the distance matrix
    heuristics = heuristics.view(distance_matrix.shape)
    
    return heuristics