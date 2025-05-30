import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand for each node
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the normalized demand for each node
    normalized_demand = cumulative_demand / demands.sum()
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal
                # Calculate the heuristic value for the edge
                heuristic_value = normalized_demand[j] - normalized_demand[i]
                # Assign the heuristic value to the corresponding edge
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix