import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    demand_vector = demands / vehicle_capacity
    
    # Calculate the heuristic for each edge based on the demand and distance
    heuristics = -distance_matrix * demand_vector
    
    # Adjust heuristics for load balancing by ensuring the sum of demands in each route does not exceed vehicle capacity
    for i in range(n):
        for j in range(n):
            if i != j:
                # Add a penalty for high demand edges to promote load balancing
                heuristics[i, j] = heuristics[i, j] - torch.max(torch.abs(demands[j] - demands[i]))
    
    # Normalize heuristics to ensure that they are in a good range for further processing
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    return heuristics