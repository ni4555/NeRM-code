import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics matrix
    # For each edge, the heuristic is the negative of the normalized demand
    # This assumes that we want to prioritize edges with lower demands
    heuristics_matrix = -normalized_demands[:, None] * normalized_demands
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    heuristics_matrix = heuristics_matrix + epsilon
    
    # Normalize the heuristics matrix to ensure that it has a range of values
    max_value = heuristics_matrix.max()
    min_value = heuristics_matrix.min()
    heuristics_matrix = (heuristics_matrix - min_value) / (max_value - min_value)
    
    return heuristics_matrix