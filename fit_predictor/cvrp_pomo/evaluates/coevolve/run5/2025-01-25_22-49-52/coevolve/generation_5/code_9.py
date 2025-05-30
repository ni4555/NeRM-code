import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Normalize the demands by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the potential for each edge based on distance and demand
    # The heuristic is defined as -distance + normalized_demand to encourage
    # shorter distances and higher demands.
    heuristic_matrix = -distance_matrix + normalized_demands
    
    # To ensure the heuristic is positive, we can add a very small positive constant
    # to all elements of the heuristic matrix.
    epsilon = 1e-6
    heuristic_matrix = torch.clamp(heuristic_matrix, min=epsilon)
    
    return heuristic_matrix