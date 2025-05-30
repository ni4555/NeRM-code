import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge based on distance and demand
    heuristic_matrix = -distance_matrix + normalized_demands
    
    # Normalize the heuristic matrix for consistent scaling
    max_value = heuristic_matrix.max()
    min_value = heuristic_matrix.min()
    normalized_heuristic_matrix = (heuristic_matrix - min_value) / (max_value - min_value)
    
    return normalized_heuristic_matrix