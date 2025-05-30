import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demand by dividing by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Normalize distance matrix by dividing by the maximum distance
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the heuristics as a combination of normalized demand and distance
    # Negative values for undesirable edges, positive for promising ones
    heuristics = normalized_distance_matrix - normalized_demands
    
    return heuristics