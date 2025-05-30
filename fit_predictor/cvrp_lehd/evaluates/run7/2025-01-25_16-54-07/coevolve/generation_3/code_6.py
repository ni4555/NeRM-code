import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    demand_total = demands.sum()
    demand_vector = demands / demand_total
    
    # Calculate savings for each edge
    savings_matrix = distance_matrix.clone()
    savings_matrix[distance_matrix == 0] = float('inf')  # No savings for the depot
    savings_matrix = savings_matrix ** 2
    
    for i in range(n):
        for j in range(n):
            if i != j:
                savings_matrix[i, j] -= 2 * distance_matrix[i, j] * demand_vector[i] * demand_vector[j]
    
    # Normalize savings matrix to get heuristics
    heuristics = savings_matrix / (2 * distance_matrix ** 2)
    heuristics[distance_matrix == 0] = 0  # No savings from the depot to itself
    
    return heuristics