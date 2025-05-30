import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load difference between each pair of nodes
    load_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    
    # Calculate the heuristic value as the negative of the load difference
    # and add a small constant to avoid zero heuristic values
    heuristic_matrix = -load_diff + 1e-6
    
    # Invert the heuristic matrix to prioritize promising edges
    # Negative values will be less desirable, positive values more desirable
    heuristic_matrix = -heuristic_matrix
    
    return heuristic_matrix