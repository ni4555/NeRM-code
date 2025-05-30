import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Create a matrix of negative infinity for the heuristics
    heuristics = torch.full_like(distance_matrix, fill_value=float('-inf'))
    
    # Calculate the maximum capacity for each vehicle (which is 1 in this normalized case)
    max_capacity = torch.ones_like(demands)
    
    # Iterate over all pairs of nodes to calculate heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Avoid considering the depot node for the heuristics
            if i == 0 or j == 0:
                continue
            
            # Calculate the potential profit of visiting node j from node i
            profit = distance_matrix[i, j] - demands[j]
            
            # Update the heuristics matrix with the potential profit if it's positive
            if profit > 0:
                heuristics[i, j] = profit
    
    # Normalize the heuristics by the total demand to make it relative to the vehicle capacity
    heuristics /= total_demand
    
    return heuristics