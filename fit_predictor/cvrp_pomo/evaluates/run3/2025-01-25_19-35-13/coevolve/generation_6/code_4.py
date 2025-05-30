import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values based on the normalized demands
    heuristics = distance_matrix - normalized_demands.unsqueeze(1) * distance_matrix
    
    # Enforce the constraint that the sum of heuristics for each node must be non-negative
    # by ensuring that the diagonal elements are the minimum value in the row
    for i in range(n):
        heuristics[i, i] = torch.min(heuristics[i, :])
    
    return heuristics