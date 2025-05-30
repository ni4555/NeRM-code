import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the demand per vehicle
    demand_per_vehicle = total_demand / demands.size(0)
    
    # Calculate the heuristic value for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic value based on the demand and distance
                heuristic_value = -distance_matrix[i][j] * demands[j] / demand_per_vehicle
                heuristic_matrix[i][j] = heuristic_value
    
    return heuristic_matrix