import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure distance_matrix and demands are on the same device and type
    distance_matrix = distance_matrix.to(demands.device).type_as(demands)
    
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Iterate over all pairs of nodes (i, j)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the cost of traveling from node i to node j
                edge_cost = distance_matrix[i, j]
                
                # Calculate the load change if this edge is taken
                load_change = demands[j] - demands[i]
                
                # Normalize the load change by the total capacity
                normalized_load_change = load_change / total_capacity
                
                # Update the heuristics matrix
                heuristics_matrix[i, j] = edge_cost - normalized_load_change
    
    return heuristics_matrix