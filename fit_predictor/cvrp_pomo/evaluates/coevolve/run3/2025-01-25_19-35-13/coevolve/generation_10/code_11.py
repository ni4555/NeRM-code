import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of all customer demands
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # The heuristic value is based on the normalized demand of the customer
                # The negative sign is used to indicate a desirable edge (smaller values are better)
                heuristic_matrix[i][j] = -normalized_demands[j]
    
    return heuristic_matrix