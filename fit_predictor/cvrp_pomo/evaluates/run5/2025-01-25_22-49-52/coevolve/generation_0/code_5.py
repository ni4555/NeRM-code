import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Calculate the average demand per customer
    average_demand = total_demand / demands.size(0)
    
    # Calculate the difference between the average demand and the actual demand for each customer
    demand_diff = demands - average_demand
    
    # Calculate the heuristic value for each edge
    # Promising edges will have positive values, undesirable edges will have negative values
    heuristics = -distance_matrix * demand_diff
    
    return heuristics