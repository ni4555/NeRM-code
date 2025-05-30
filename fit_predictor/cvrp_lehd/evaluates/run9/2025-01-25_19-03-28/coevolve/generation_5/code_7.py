import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate through the distance matrix to calculate heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value for the edge from i to j
                # Here we use a simple heuristic that considers both distance and demand
                heuristics[i, j] = -distance_matrix[i, j] + normalized_demands[i] * normalized_demands[j]
            else:
                # The edge from a node to itself should not be considered
                heuristics[i, j] = float('-inf')
    
    return heuristics