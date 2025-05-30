import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the load for each customer
    load = demands / total_capacity
    
    # Iterate over each customer
    for i in range(1, n):
        # Calculate the difference in load for each possible edge
        load_diff = load[i] - load
        
        # Calculate the heuristic value for each edge
        heuristic_matrix[i, :] = -distance_matrix[i, :] * load_diff
    
    # Normalize the heuristic matrix to ensure that it contains negative values for undesirable edges
    # and positive values for promising ones
    heuristic_matrix = (heuristic_matrix - heuristic_matrix.min()) / (heuristic_matrix.max() - heuristic_matrix.min())
    
    return heuristic_matrix