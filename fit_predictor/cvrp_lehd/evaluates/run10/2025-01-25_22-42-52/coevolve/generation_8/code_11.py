import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the cost of each edge
    cost_matrix = distance_matrix * normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    cost_matrix = cost_matrix / (cost_matrix + epsilon)
    
    # Subtract from 1 to get the heuristic values
    heuristics = 1 - cost_matrix
    
    return heuristics