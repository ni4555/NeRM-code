import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands
    total_demand = demands.sum()
    
    # Normalize the demands to represent the fraction of vehicle capacity used by each customer
    normalized_demands = demands / total_demand
    
    # Initialize a tensor of the same shape as the distance matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    # For each customer (i), calculate the potential cost of visiting from the depot (0)
    for i in range(1, len(demands)):
        # Calculate the heuristic value for the edge from depot to customer i
        heuristics[0, i] = -distance_matrix[0, i] * normalized_demands[i]
    
    # For each customer (i), calculate the heuristic value for the edge from customer i to the depot (0)
    for i in range(1, len(demands)):
        # Calculate the heuristic value for the edge from customer i to depot (0)
        heuristics[i, 0] = -distance_matrix[i, 0] * normalized_demands[i]
    
    return heuristics