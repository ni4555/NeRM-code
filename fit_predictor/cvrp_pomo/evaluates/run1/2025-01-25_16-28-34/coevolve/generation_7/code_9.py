import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum() / demands.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Compute the heuristic value for each edge based on the demand and distance
    # For simplicity, let's use a linear function: heuristic = -distance - demand
    heuristic_matrix = -distance_matrix - demands.unsqueeze(1)
    
    # Normalize the heuristic matrix by the vehicle capacity to encourage balanced load distribution
    heuristic_matrix /= vehicle_capacity
    
    # Adjust the heuristic matrix to ensure positive values for promising edges
    heuristic_matrix = torch.clamp(heuristic_matrix, min=-1e-6, max=float('inf'))
    
    return heuristic_matrix