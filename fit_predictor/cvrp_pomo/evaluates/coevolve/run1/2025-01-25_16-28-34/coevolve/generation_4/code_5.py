import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_capacity
    
    # Calculate the normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the initial heuristic values based on normalized demand and distance
    # We will use a simple formula that combines the normalized demand and normalized distance
    heuristic_values = -normalized_demands * normalized_distance_matrix
    
    # Adjust the heuristic values to ensure they are negative for undesirable edges
    # We can use a simple threshold to convert positive values to negative ones
    threshold = 0.5
    heuristic_values[heuristic_values > threshold] = -heuristic_values[heuristic_values > threshold]
    
    return heuristic_values