import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the cumulative demand along each route
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative distance along each route
    cumulative_distance = torch.cumsum(distance_matrix, dim=0)
    
    # Nearest neighbor heuristic: Assign a negative score to edges that exceed vehicle capacity
    for i in range(1, n):
        for j in range(i + 1, n):
            if cumulative_demand[j] - cumulative_demand[i - 1] > 1.0:  # Exceeds capacity
                heuristic_matrix[i, j] = -1.0
                heuristic_matrix[j, i] = -1.0
    
    # Adjust the heuristic matrix to ensure that the depot is the best starting point
    for i in range(1, n):
        for j in range(1, n):
            if heuristic_matrix[i, j] == 0 and heuristic_matrix[i, 0] == 0 and heuristic_matrix[0, j] == 0:
                heuristic_matrix[i, j] = cumulative_distance[j, 0] - cumulative_distance[i, 0]
    
    return heuristic_matrix