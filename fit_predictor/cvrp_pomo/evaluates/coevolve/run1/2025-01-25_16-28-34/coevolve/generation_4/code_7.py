import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate normalized distance matrix
    normalized_distance_matrix = distance_matrix / distance_matrix.sum(axis=0)
    
    # Calculate the initial heuristics based on normalized demand and distance
    heuristics = normalized_distance_matrix * normalized_demands
    
    return heuristics