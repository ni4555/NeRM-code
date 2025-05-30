import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix and demands are on the same device and of the same dtype
    distance_matrix = distance_matrix.to(demands.device).to(demands.dtype)
    demands = demands.to(distance_matrix.device).to(distance_matrix.dtype)
    
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate heuristic values based on normalized demands
    # A simple heuristic could be the negative of the demand (less desirable edges have higher costs)
    heuristic_matrix = -normalized_demands
    
    # Incorporate distance factor into the heuristic (more distant edges have higher costs)
    heuristic_matrix += distance_matrix
    
    # Adjust the heuristic matrix to ensure positive values for promising edges
    # We use a small positive constant to avoid division by zero
    epsilon = 1e-6
    heuristic_matrix = torch.clamp(heuristic_matrix, min=epsilon)
    
    return heuristic_matrix