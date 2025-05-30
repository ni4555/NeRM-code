import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize by total vehicle capacity
    total_demand = torch.sum(demands)
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_demand
    
    # Initialize a tensor with zeros with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop through each edge to determine if it is promising or not
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Check if the edge is not the diagonal ( depot to itself )
            if i != j:
                # Calculate the potential value of this edge
                edge_value = distance_matrix[i, j] - normalized_demands[i]
                
                # Assign the value to the corresponding edge in the heuristics matrix
                heuristics[i, j] = edge_value
    
    return heuristics