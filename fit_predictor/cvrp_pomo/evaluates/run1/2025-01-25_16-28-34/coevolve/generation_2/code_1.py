import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the sum of all demands to get demand per unit of capacity
    demand_per_capacity = demands / demands.sum()
    
    # Calculate the sum of demands for each node (including the depot)
    sum_of_demands = (demands + demand_per_capacity).cumsum(dim=0)
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop through each node to calculate the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Exclude the diagonal (self-loops)
                # Calculate the heuristic based on the sum of demands and distance
                heuristics[i, j] = -sum_of_demands[j] + sum_of_demands[i]
    
    return heuristics