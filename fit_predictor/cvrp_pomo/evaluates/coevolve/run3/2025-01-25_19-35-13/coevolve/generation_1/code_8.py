import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix
    distance_matrix = distance_matrix / distance_matrix.max()
    
    # Calculate the heuristic based on the normalized distance and demand
    # The heuristic is a combination of the normalized distance and the negative demand
    # Negative demand is used to prioritize edges with lower demand
    heuristics = distance_matrix - demands
    
    return heuristics