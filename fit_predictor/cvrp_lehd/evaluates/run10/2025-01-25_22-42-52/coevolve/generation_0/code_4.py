import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate through the distance matrix
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Skip the diagonal
                # Calculate the demand for the current edge
                current_demand = demands[i] + demands[j]
                
                # Normalize the demand by the total vehicle capacity
                normalized_demand = current_demand / total_demand
                
                # Calculate the heuristic value
                heuristic_value = normalized_demand * distance_matrix[i, j]
                
                # Assign the heuristic value to the edge
                heuristics[i, j] = heuristic_value
    
    return heuristics