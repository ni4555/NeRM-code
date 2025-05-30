import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the negative of the demand to use as a heuristic
    negative_demands = -normalized_demands
    
    # Create a matrix of ones to represent all possible edges
    edge_matrix = torch.ones_like(distance_matrix)
    
    # Subtract the negative demand from the edge distances to create a heuristic
    edge_matrix = edge_matrix * distance_matrix - negative_demands.unsqueeze(1)
    
    # Replace any negative values with zeros to indicate undesirable edges
    edge_matrix[edge_matrix < 0] = 0
    
    return edge_matrix