import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize the demands vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Iterate over all edges in the graph (excluding the diagonal)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the edge heuristics based on distance and demand
                # The heuristic is negative to indicate that this edge is initially undesirable
                heuristics_matrix[i, j] = -distance_matrix[i, j] - normalized_demands[i] - normalized_demands[j]
    
    return heuristics_matrix