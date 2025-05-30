import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands vector includes the depot's demand (which should be 0)
    demands = torch.cat([torch.zeros(1), demands])
    
    # Calculate the cumulative demand matrix
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the cumulative distance matrix
    cumulative_distance = torch.triu(distance_matrix)  # Triangular matrix of distances (excluding diagonal)
    
    # Create an initial heuristic matrix where each edge's weight is negative
    heuristics = -cumulative_distance
    
    # For each node, add the difference between the cumulative demand and the capacity (1 unit)
    # to make more promising edges (edges with lower cumulative demand difference)
    for i in range(len(demands)):
        heuristics[i, i + 1:] = heuristics[i, i + 1:] + (cumulative_demand[i + 1:] - cumulative_demand[i])
    
    # Return the heuristic matrix
    return heuristics