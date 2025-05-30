import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot = 0
    total_capacity = demands.sum()
    
    # Calculate the maximum demand that can be covered by a single vehicle
    max_demand_per_vehicle = total_capacity / n
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristic value for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the demand and distance
                heuristic_value = demands[j] / (distance_matrix[i, j] + 1e-8)
                
                # Normalize the heuristic value by the maximum demand per vehicle
                heuristic_value /= max_demand_per_vehicle
                
                # Assign the heuristic value to the edge
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix