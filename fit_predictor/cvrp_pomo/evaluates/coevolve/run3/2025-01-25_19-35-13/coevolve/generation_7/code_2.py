import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Initialize potential function matrix
    potential_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate edge weights based on normalized demands, distance, and road quality (assumed as 1 for simplicity)
    for i in range(n):
        for j in range(n):
            if i != j:  # Exclude self-loops
                potential_matrix[i, j] = normalized_demands[i] + normalized_demands[j] + distance_matrix[i, j]
    
    # Handle division by zero by setting a small threshold
    threshold = 1e-8
    potential_matrix[potential_matrix <= threshold] = -threshold
    
    return potential_matrix