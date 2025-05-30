import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand as a reference for normalization
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Create a matrix of negative values (initially undesirable edges)
    heuristics_matrix = -torch.ones_like(distance_matrix)
    
    # Calculate the heuristics based on normalized demands
    # The heuristic for an edge is the negative of the normalized demand of the customer at the end of the edge
    heuristics_matrix[1:, :] = -normalized_demands[1:]
    
    return heuristics_matrix