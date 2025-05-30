import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values using the Chebyshev distance metric
    # This is a simple heuristic that considers the maximum of the distance and the demand
    heuristic_matrix = torch.clamp(distance_matrix + demands, min=0)
    
    return heuristic_matrix