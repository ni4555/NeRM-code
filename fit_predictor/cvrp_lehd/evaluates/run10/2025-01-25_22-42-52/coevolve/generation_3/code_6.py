import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # Calculate the heuristic value for each edge
    # This heuristic is a simple combination of distance and demand
    # where we penalize longer distances and higher demands
    heuristic_matrix = -distance_matrix + (1 - normalized_demands) * distance_matrix
    
    return heuristic_matrix