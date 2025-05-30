import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands are on the same device
    demands = demands.to(distance_matrix.device)
    
    # Normalize demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Create a matrix with 0s, indicating no direct demand-related cost
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Add the demand-based heuristic for each edge
    heuristic_matrix = heuristic_matrix - normalized_demands.unsqueeze(0).expand_as(distance_matrix)
    
    # Optionally, add other heuristics such as distance-based penalties or rewards
    
    return heuristic_matrix