import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand
    total_demand = demands.sum()
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_demand
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the distance and demand
                heuristic_value = -distance_matrix[i][j] + normalized_demands[j]
                heuristic_matrix[i][j] = heuristic_value
    
    return heuristic_matrix