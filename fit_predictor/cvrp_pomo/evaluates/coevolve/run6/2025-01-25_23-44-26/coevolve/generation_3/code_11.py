import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are already normalized by the total vehicle capacity
    n = distance_matrix.shape[0]
    
    # Initialize a matrix of zeros with the same shape as distance_matrix
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand to normalize the distances
    total_demand = demands.sum()
    
    # Normalize the distance matrix by the total demand
    normalized_distance_matrix = distance_matrix / total_demand
    
    # For each edge, calculate the heuristic value based on the normalized demand
    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude the depot node
                # A simple heuristic: the smaller the normalized distance, the more promising the edge
                heuristics_matrix[i, j] = -normalized_distance_matrix[i, j]
    
    return heuristics_matrix