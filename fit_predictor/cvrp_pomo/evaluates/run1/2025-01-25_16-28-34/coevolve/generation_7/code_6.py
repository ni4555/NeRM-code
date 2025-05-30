import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Calculate the sum of the demands divided by the total capacity
    # This will be used to normalize the cost function
    demand_sum = demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Loop through all possible edges (except the diagonal)
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:  # Skip the diagonal as it represents the distance from the depot to itself
                # Calculate the cost of traveling from customer i to customer j
                cost = distance_matrix[i, j]
                
                # Normalize the cost based on the demands of the customers
                # Negative cost if it is cheaper to visit customer j from i than the other way around
                if demands[j] > demands[i]:
                    cost *= -1
                else:
                    cost *= 1
                
                # Adjust the cost based on the load balance
                # If the load after visiting customer j exceeds the vehicle capacity, add a penalty
                if (demands[i] + demands[j]) > total_capacity:
                    cost += 1000  # This is a large penalty to avoid this edge
                
                # Normalize the cost by the sum of demands
                heuristics[i, j] = cost / demand_sum
    
    return heuristics