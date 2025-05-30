import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    
    # Calculate the cumulative sum of demands to identify the points where the vehicle must return
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the number of vehicles needed based on the total demand and vehicle capacity
    vehicle_count = torch.ceil(cumulative_demand / demands[0]).int()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # For each customer, set the heuristics for the edge from the previous customer
    for i in range(1, n):
        heuristics[i, i-1] = -vehicle_count[i]
    
    # For the last customer, set the heuristics for the edge to the depot
    heuristics[n-1, 0] = -vehicle_count[n-1]
    
    return heuristics