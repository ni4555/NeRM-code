import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum() / n  # Assuming the total demand is evenly distributed among vehicles
    
    # Normalize demands by vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the normalized demands
    for i in range(n):
        for j in range(n):
            if i != j:
                # For each edge (i, j), calculate the heuristic value
                # This is a simple heuristic that considers the demand of the customer node
                heuristic_value = -normalized_demands[j]
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix