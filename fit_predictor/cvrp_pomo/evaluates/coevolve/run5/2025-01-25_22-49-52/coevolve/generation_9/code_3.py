import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to ensure consistency
    min_distance = distance_matrix.min()
    max_distance = distance_matrix.max()
    normalized_distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Normalize the demands to ensure they sum up to the vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Calculate the initial heuristic values based on normalized demands
    # and a simple heuristic that takes into account the distance
    heuristic_matrix = normalized_distance_matrix * normalized_demands.unsqueeze(1)
    
    # Add a penalty for edges leading to the depot to discourage starting/ending at the depot
    depot_penalty = -1
    heuristic_matrix[torch.arange(distance_matrix.size(0)), torch.arange(distance_matrix.size(0))] = depot_penalty
    
    return heuristic_matrix