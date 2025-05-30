import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the maximum possible demand for any edge
    max_demand = normalized_demands.max()
    
    # Calculate the negative potential for each edge
    # Negative potential is high for edges with high demand and low distance
    negative_potential = -normalized_demands * distance_matrix
    
    # Normalize the negative potential by the maximum demand
    normalized_negative_potential = negative_potential / max_demand
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    normalized_negative_potential = normalized_negative_potential + epsilon
    
    # Calculate the heuristic values as the negative of the normalized potential
    heuristics = -normalized_negative_potential
    
    return heuristics