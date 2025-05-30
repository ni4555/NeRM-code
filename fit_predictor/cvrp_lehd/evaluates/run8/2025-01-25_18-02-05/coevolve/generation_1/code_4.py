import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum distance in the matrix to use as a base for normalization
    max_distance = torch.max(distance_matrix)
    
    # Calculate the total demand to normalize the demands vector
    total_demand = torch.sum(demands)
    
    # Normalize the demands vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristics matrix with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over all possible edges (excluding the diagonal and self-loops)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value based on the normalized demand and distance
                heuristics[i, j] = normalized_demands[i] * normalized_demands[j] * (distance_matrix[i, j] / max_distance)
    
    return heuristics