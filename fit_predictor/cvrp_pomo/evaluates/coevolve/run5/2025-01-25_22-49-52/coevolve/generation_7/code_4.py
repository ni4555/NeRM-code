import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize the distance matrix
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Compute the heuristic value for each edge
    # The heuristic function is a combination of normalized distance and normalized demand
    # Negative values indicate undesirable edges, positive values indicate promising ones
    heuristics = -normalized_distance_matrix + normalized_demands
    
    return heuristics