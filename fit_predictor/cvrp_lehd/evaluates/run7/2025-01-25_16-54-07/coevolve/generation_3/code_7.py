import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the savings for each edge
    savings = distance_matrix.clone()
    for i in range(n):
        for j in range(n):
            if i != j:
                savings[i, j] = demands[i] + demands[j] - distance_matrix[i, j]
    
    # Normalize savings by the sum of demands to get a relative measure
    savings /= demands.sum()
    
    # Add a penalty for edges that exceed the time window or are not cost-effective
    # Assuming there's a threshold beyond which an edge is not cost-effective
    threshold = 1.0  # Example threshold
    savings[distance_matrix > threshold] = -float('inf')
    
    return savings