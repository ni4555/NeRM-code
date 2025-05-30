import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the heuristic values for each edge
    # Using the nearest neighbor heuristic, we assume that the distance to the nearest
    # customer is a good indicator of the edge's potential value
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each customer (ignoring the depot), find the minimum distance to any other customer
    for i in range(1, len(demands)):
        min_distance = distance_matrix[i, 1:].min(dim=1)[0]
        heuristics[i, 1:] = -min_distance
    
    # For the depot, assign a heuristic of 0 to all edges (excluding itself)
    heuristics[0, 1:] = 0
    
    # Normalize the heuristic values based on the total demand
    heuristics /= total_demand
    
    return heuristics