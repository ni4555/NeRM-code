import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assumed to be 1 for normalization purposes)
    total_capacity = 1.0
    
    # Create a matrix to store the heuristics
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Normalize demands and calculate the negative of the inverse of the demand divided by the capacity
    # This heuristic prioritizes including edges with lower demand
    heuristic_matrix += -torch.div(torch.abs(demands), total_capacity)
    
    # Add a small constant to avoid zero division
    heuristic_matrix += 1e-6
    
    return heuristic_matrix