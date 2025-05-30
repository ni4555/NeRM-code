import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each node
    demand_sum = demands.sum(dim=0)
    
    # Calculate the maximum demand that can be handled by a vehicle
    max_demand = demands[0]  # Assuming the depot demand is the vehicle capacity
    
    # Calculate the difference between the sum of demands and the vehicle capacity
    demand_diff = demand_sum - max_demand
    
    # Calculate the heuristic values based on the difference
    # Promising edges will have positive values, undesirable edges will have negative values
    heuristics = torch.where(demand_diff > 0, -demand_diff, torch.zeros_like(demand_diff))
    
    return heuristics