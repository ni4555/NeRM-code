import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demands_normalized = demands / demands.sum()
    
    # Initialize the heuristic matrix with large negative values
    heuristics = -torch.ones_like(distance_matrix)
    
    # Calculate heuristic based on distance and demand ratio
    heuristics[distance_matrix != 0] = -distance_matrix[distance_matrix != 0] * demands_normalized[distance_matrix != 0]
    
    # Apply a normalization to ensure positive values
    heuristics = heuristics - heuristics.min()
    
    return heuristics