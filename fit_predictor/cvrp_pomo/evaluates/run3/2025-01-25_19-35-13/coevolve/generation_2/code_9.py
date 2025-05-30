import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Create a matrix with the same shape as the distance matrix initialized to zero
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value for the edge (i, j)
                heuristics_matrix[i, j] = -distance_matrix[i, j] + normalized_demands[i] * normalized_demands[j]
    
    return heuristics_matrix