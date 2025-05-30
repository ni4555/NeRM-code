import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device and have the same dtype
    distance_matrix = distance_matrix.to(demands.device).float()
    demands = demands.to(distance_matrix.device).float()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the sum of demands for each customer
    demand_sum = demands.sum()
    
    # Iterate over all pairs of nodes
    for i in range(len(demands)):
        for j in range(len(demands)):
            if i != j:
                # Calculate the demand left if this customer is visited
                demand_left = demand_sum - demands[i]
                
                # If the demand left is less than or equal to the vehicle capacity
                if demand_left <= 1.0:
                    # The heuristic value is the distance to the next customer
                    heuristics[i, j] = distance_matrix[i, j]
                else:
                    # Otherwise, it's an undesirable edge with a negative heuristic
                    heuristics[i, j] = -float('inf')
    
    return heuristics