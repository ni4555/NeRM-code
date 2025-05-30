import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it is a single value for all vehicles)
    vehicle_capacity = 1.0
    
    # Normalize demands to the range [0, 1] using the vehicle capacity
    normalized_demands = demands / demands.sum()
    
    # Initialize a matrix of the same shape as distance_matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the potential cost of visiting each edge
    # This heuristic could be based on the normalized demand of the destination customer
    # A simple heuristic is to penalize high demands or high distances
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate a heuristic value for each edge
                # Here we use the normalized demand of the customer as the heuristic value
                # This assumes that higher demand leads to a more promising edge
                heuristic_value = normalized_demands[j]
                
                # Update the heuristic matrix with the calculated heuristic value
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix