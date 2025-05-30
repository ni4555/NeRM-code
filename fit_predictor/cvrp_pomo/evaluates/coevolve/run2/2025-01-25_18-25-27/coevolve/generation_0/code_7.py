import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum demand that a single vehicle can carry
    vehicle_capacity = demands.sum()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the difference between demands and vehicle capacity
    demand_diff = demands - vehicle_capacity
    
    # For each customer (excluding the depot), calculate the potential profit of visiting the customer
    # which is the difference between the vehicle capacity and the demand of the customer
    heuristics[demand_diff > 0] = vehicle_capacity - demand_diff[demand_diff > 0]
    
    return heuristics