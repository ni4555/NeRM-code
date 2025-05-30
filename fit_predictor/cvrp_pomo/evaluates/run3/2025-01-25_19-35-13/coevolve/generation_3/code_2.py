import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix
    max_distance = distance_matrix.max()
    min_distance = distance_matrix.min()
    normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the potential function, combining normalized distances and demands
    potential = normalized_distance_matrix * normalized_demands
    
    # Use an epsilon value to prevent division by zero
    epsilon = 1e-8
    
    # Calculate the heuristics values
    heuristics = potential / (potential + epsilon)
    
    return heuristics