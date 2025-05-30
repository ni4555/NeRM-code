import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to the range of [0, 1]
    demands_normalized = demands / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the normalized demands
    heuristics = -distance_matrix * demands_normalized
    
    # Apply a penalty for edges that lead to overloading
    # Assuming that the maximum capacity is 1 for simplicity
    heuristics = heuristics + (1 - demands_normalized)
    
    return heuristics