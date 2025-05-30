import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix to calculate heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value for each edge
                # Here we use a simple heuristic that considers the demand at customer j
                # and the distance from customer i to j. This is a naive approach and
                # can be replaced with more sophisticated heuristics.
                heuristics[i, j] = -normalized_demands[j] + distance_matrix[i, j]
    
    return heuristics