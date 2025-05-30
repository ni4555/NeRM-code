import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total demand
    total_demand = demands.sum()
    # Calculate the average demand per customer
    average_demand = total_demand / n
    # Create a heuristics matrix initialized with negative infinity
    heuristics = -torch.full((n, n), float('inf'))
    
    # Calculate heuristics for edges between the depot (0) and customers
    heuristics[0, 1:] = -distance_matrix[0, 1:]
    # Calculate heuristics for edges between customers
    for i in range(1, n):
        for j in range(i + 1, n):
            # The heuristic for an edge is the difference in demands between the two customers
            # and a penalty for the distance
            heuristics[i, j] = demands[i] - demands[j] - distance_matrix[i, j]
            heuristics[j, i] = demands[j] - demands[i] - distance_matrix[i, j]
    # Normalize the heuristics by the average demand
    heuristics = heuristics + average_demand
    return heuristics