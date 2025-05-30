import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to get the fraction of capacity each customer requires
    normalized_demands = demands / demands.sum()
    
    # Initialize a matrix to hold the heuristic values, set to 0 initially
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Iterate over each node to calculate the heuristic values
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # If it's not the same node, calculate the heuristic value
            if i != j:
                # Calculate the heuristic value as the negative of the normalized demand
                # This assumes that lower demand values are better, hence the negative sign
                heuristic_value = -normalized_demands[j]
                # Set the heuristic value for the edge from i to j
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix