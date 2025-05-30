import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands to normalize the individual demands
    total_demand = torch.sum(demands)
    
    # Normalize the demands vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Create a vector that is the same shape as the distance matrix with all values set to -1
    heuristics = -torch.ones_like(distance_matrix)
    
    # Iterate over the matrix to assign heuristics based on normalized demands
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Exclude the depot node from the calculations
                # The heuristic for an edge is the normalized demand of the customer node
                heuristics[i, j] = normalized_demands[j]
    
    return heuristics