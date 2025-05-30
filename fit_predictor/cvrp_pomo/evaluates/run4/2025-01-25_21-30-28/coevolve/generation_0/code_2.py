import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for all customers
    total_demand = demands.sum()
    
    # Create a vector that indicates if the demand at each node exceeds the vehicle capacity
    # (total_demand / len(demands)) represents the threshold
    demand_exceeds_capacity = demands > (total_demand / demands.size(0))
    
    # Initialize the heuristics matrix with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i == 0 and j != 0:  # Only consider edges leaving the depot
                if demand_exceeds_capacity[j]:  # If the customer's demand is high, mark it as promising
                    heuristics[i, j] = 1.0
                else:  # Otherwise, it's not promising
                    heuristics[i, j] = -1.0
    return heuristics