import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate the cost matrix which includes demands
    cost_matrix = distance_matrix.clone()
    cost_matrix = torch.cat([cost_matrix, demands.unsqueeze(0)], dim=0)
    cost_matrix = torch.cat([cost_matrix, torch.zeros(n, 1)], dim=1)
    cost_matrix = cost_matrix + torch.transpose(cost_matrix, 0, 1)
    cost_matrix = cost_matrix - torch.diag(torch.diag(cost_matrix))
    
    # Calculate the initial heuristic values
    heuristic_matrix = torch.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):
            # Prioritize edges that have lower distance and lower demand difference
            heuristic_matrix[i, j] = -distance_matrix[i, j] - abs(demands[i] - demands[j])
    
    # Apply problem-specific local search to enhance the heuristic values
    for i in range(n):
        for j in range(i+1, n):
            # Example: Improve heuristic by considering vehicle capacity
            # (This is a placeholder for a more complex local search)
            heuristic_matrix[i, j] += (demands[i] + demands[j]) / (2 * demands[i])
    
    # Normalize the heuristic matrix
    heuristic_matrix = heuristic_matrix - heuristic_matrix.min()
    heuristic_matrix = (heuristic_matrix / heuristic_matrix.max()).unsqueeze(0)
    
    return heuristic_matrix