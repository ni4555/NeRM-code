import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Find the nearest neighbor for each customer and calculate the heuristic
    for i in range(1, n):  # Start from the first customer (index 1)
        min_distance = torch.min(distance_matrix[i])
        heuristics[i] = -min_distance
    
    # Demand-driven heuristic: Add the cumulative demand of the route
    cumulative_demand = demands[1:].cumsum()
    cumulative_demand = torch.cat([torch.zeros(1), cumulative_demand])
    
    # Add the cumulative demand as a positive heuristic
    heuristics[1:] += cumulative_demand
    
    # Adjust the heuristic to be positive for promising edges and negative for others
    heuristics = heuristics.clamp(min=0, max=float('inf'))
    heuristics[heuristics <= 0] = float('-inf')
    
    return heuristics