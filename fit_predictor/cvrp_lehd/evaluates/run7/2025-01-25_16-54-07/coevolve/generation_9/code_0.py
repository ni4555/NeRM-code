import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cumulative demand for each potential route
    cumulative_demand = demands.cumsum(dim=0)
    
    # Calculate the difference between the cumulative demand and the maximum capacity
    demand_excess = cumulative_demand - demands[0]  # Excess demand for each node after the depot
    
    # Calculate the heuristics based on the difference in demand
    heuristics = -demand_excess
    
    # Normalize the heuristics to ensure they are within the range of the distance matrix
    heuristics = heuristics / (distance_matrix.max() + 1)
    
    return heuristics