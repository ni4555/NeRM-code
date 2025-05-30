import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    adjusted_demands = demands / total_capacity
    
    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the potential benefit of each edge
    for i in range(1, n):  # Skip the depot node (index 0)
        for j in range(i + 1, n):  # Skip symmetric edges
            heuristics[i, j] = -distance_matrix[i, j]  # Negative cost for undesirable edges
            heuristics[j, i] = -distance_matrix[i, j]
            
            # Adjust the heuristics based on customer demand
            if adjusted_demands[i] + adjusted_demands[j] <= 1.0:
                heuristics[i, j] += 1  # Positive cost for promising edges
                heuristics[j, i] += 1

    return heuristics