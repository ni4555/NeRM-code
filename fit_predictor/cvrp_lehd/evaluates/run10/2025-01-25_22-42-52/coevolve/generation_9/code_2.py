import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize customer demands
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Compute the heuristics
    # We use a simple heuristic based on normalized demand, here we use negative of it for vectorization
    heuristic_matrix = -normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)
    
    return heuristic_matrix