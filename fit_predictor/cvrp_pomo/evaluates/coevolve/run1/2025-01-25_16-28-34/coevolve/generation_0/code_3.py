import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative sum of demands to find the first point of overflow
    cumsum_demands = torch.cumsum(demands, dim=0)
    
    # Initialize a tensor of the same shape as distance_matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix to populate heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # If the next customer exceeds vehicle capacity, set this edge heuristic to -1
            if cumsum_demands[j] > 1:
                heuristics[i, j] = -1
            # Otherwise, set it to the distance
            else:
                heuristics[i, j] = distance_matrix[i, j]
    
    return heuristics