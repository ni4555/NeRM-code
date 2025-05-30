import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are already normalized by the total vehicle capacity
    total_capacity = demands.sum()
    demand_vector = demands / total_capacity
    
    # Initialize heuristics matrix with zeros
    n = distance_matrix.shape[0]
    heuristics_matrix = torch.zeros(n, n, dtype=torch.float32)
    
    # Compute the heuristics for each edge based on demand
    for i in range(n):
        for j in range(i+1, n):  # Skip the diagonal to avoid double counting
            if demands[j] != 0:  # Skip if it's the depot or the demand is zero
                heuristics_matrix[i, j] = -demand_vector[i] * demand_vector[j] * distance_matrix[i, j]
                heuristics_matrix[j, i] = heuristics_matrix[i, j]  # Symmetry
    
    return heuristics_matrix