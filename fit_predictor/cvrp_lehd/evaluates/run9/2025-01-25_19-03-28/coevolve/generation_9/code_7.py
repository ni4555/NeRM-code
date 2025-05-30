import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the negative weighted distance for each edge
    negative_weighted_distance = -distance_matrix
    
    # Apply normalized demand to the negative weighted distance
    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude the depot from the calculation
                # Update the heuristic value for each edge
                heuristic_matrix[i, j] = negative_weighted_distance[i, j] * normalized_demands[i]
    
    return heuristic_matrix