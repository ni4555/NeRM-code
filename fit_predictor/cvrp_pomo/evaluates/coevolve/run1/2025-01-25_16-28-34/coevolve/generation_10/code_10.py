import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the maximum load per vehicle as a fraction of total demand
    max_load = 1.0 / demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop through each pair of nodes to calculate the heuristic
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the edge heuristic as the negative of the distance
            # multiplied by the load factor, adjusted by the demand difference
            edge_heuristic = -distance_matrix[i, j] * max_load
            if demands[i] != demands[j]:
                edge_heuristic += demands[i] * (demands[i] - demands[j])
            
            # Update the heuristics matrix
            heuristics[i, j] = edge_heuristic
            heuristics[j, i] = edge_heuristic  # The matrix is symmetric
    
    return heuristics