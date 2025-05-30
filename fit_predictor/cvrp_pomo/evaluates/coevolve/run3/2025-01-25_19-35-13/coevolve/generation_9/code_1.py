import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance from the depot to all customers
    total_depot_to_customers = torch.sum(distance_matrix[:, 1:], dim=1)
    
    # Calculate the total distance from each customer to the depot
    total_customers_to_depot = torch.sum(distance_matrix[1:, :], dim=0)
    
    # Calculate the total demand of each customer
    total_demand = torch.sum(demands[1:])
    
    # Calculate the negative heuristic based on the total demand and the total distance
    # The heuristic is negative because we want to minimize the cost
    heuristics = -1 * (total_depot_to_customers + total_customers_to_depot - total_demand)
    
    # Add a large negative value for the diagonal to avoid including the depot in the route
    heuristics += torch.full_like(heuristics, fill_value=-float('inf')) * torch.eye(len(heuristics), device=heuristics.device)
    
    return heuristics