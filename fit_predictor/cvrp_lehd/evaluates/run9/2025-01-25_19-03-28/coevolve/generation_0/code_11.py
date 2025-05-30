import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Number of nodes
    n = distance_matrix.shape[0]
    
    # Normalize demands so that they sum up to the vehicle's capacity (which is 1 in this case)
    normalized_demands = demands / demands.sum()
    
    # Initialize a matrix to hold heuristic values, with high negative values for the edges that are undesirable
    heuristic_matrix = -torch.ones_like(distance_matrix)
    
    # For each edge, calculate the cumulative demand when visiting the next customer
    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if adding this customer would exceed the vehicle's capacity
                if (heuristic_matrix[i, j] + demands[j]) > 1.0:
                    continue  # Skip this edge
                
                # Calculate the score for the edge
                score = -distance_matrix[i, j] + (1.0 - (heuristic_matrix[i, j] + demands[j]))
                heuristic_matrix[i, j] = score
    
    return heuristic_matrix