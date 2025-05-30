import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are already normalized by the total vehicle capacity
    n = demands.size(0)
    
    # Initialize heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the maximum distance between any two points in the matrix
    max_distance = torch.max(distance_matrix)
    
    # Calculate the maximum demand
    max_demand = torch.max(demands)
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / demands.sum()
    
    # For each customer, compute the heuristic based on distance and demand
    for i in range(1, n):  # Skip the depot node
        for j in range(n):
            if i != j:
                # Calculate heuristic as a combination of distance and normalized demand
                # Negative values for undesirable edges, positive values for promising ones
                heuristics[i, j] = -distance_matrix[i, j] - max_distance * normalized_demands[j]
    
    return heuristics