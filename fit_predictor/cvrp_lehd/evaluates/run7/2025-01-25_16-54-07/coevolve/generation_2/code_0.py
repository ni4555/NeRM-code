import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a matrix of negative values for all edges
    heuristics_matrix = -torch.ones_like(distance_matrix)
    
    # Calculate the potential value for each edge
    # Potential value = distance to customer * demand ratio
    potential_values = distance_matrix * normalized_demands
    
    # Set the potential values into the heuristics matrix
    heuristics_matrix += potential_values
    
    return heuristics_matrix