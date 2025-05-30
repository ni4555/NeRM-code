import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands are normalized by the total vehicle capacity
    # Assuming the total vehicle capacity is 1 for normalization purposes
    total_capacity = 1.0
    demands = demands / total_capacity
    
    # Calculate the sum of demands for each node
    demand_sum = demands.sum(dim=1)
    
    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each pair of nodes (i, j), calculate the heuristic value
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Compute the heuristic value based on the difference in demand
                heuristics[i, j] = demands[i] - demands[j]
            else:
                # The heuristic value from a node to itself should be negative
                heuristics[i, j] = -1e9
    
    return heuristics