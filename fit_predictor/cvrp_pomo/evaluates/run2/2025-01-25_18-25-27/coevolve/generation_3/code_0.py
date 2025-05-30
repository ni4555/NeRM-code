import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative distance to make longer distances more undesirable
    negative_distance_matrix = -distance_matrix
    
    # Calculate the heuristic values based on the negative distance and the normalized demands
    heuristic_matrix = negative_distance_matrix + demands
    
    # Normalize the heuristic matrix to ensure values are between -1 and 1
    min_val = heuristic_matrix.min()
    max_val = heuristic_matrix.max()
    heuristic_matrix = (heuristic_matrix - min_val) / (max_val - min_val)
    
    return heuristic_matrix