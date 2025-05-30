import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for the depot node
    total_capacity = torch.sum(demands)
    normalized_demand = demands / total_capacity
    
    # Initialize a matrix of zeros with the same shape as the distance matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics using the normalized demands and distance matrix
    # This is a simple heuristic based on the normalized demand and distance
    heuristics_matrix = -distance_matrix * (normalized_demand - 1)
    
    # Apply a normalization technique to ensure all values are within a certain range
    # (e.g., between 0 and 1), here we use the Min-Max normalization
    min_val = torch.min(heuristics_matrix)
    max_val = torch.max(heuristics_matrix)
    heuristics_matrix = (heuristics_matrix - min_val) / (max_val - min_val)
    
    return heuristics_matrix