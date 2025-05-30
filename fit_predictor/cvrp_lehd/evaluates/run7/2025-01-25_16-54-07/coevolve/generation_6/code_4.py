import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to get the relative demand at each node
    normalized_demands = demands / demands.sum()
    
    # Calculate cumulative demand mask
    cumulative_demand_mask = torch.cumsum(normalized_demands, dim=0)
    
    # Calculate the cumulative cost of each edge, considering the cumulative demand
    cumulative_cost = cumulative_demand_mask * distance_matrix
    
    # Calculate the edge feasibility mask by subtracting the total vehicle capacity
    # from the cumulative cost to see how much we can increase the load
    edge_feasibility_mask = cumulative_cost - demands
    
    # Prioritize edges by their feasibility and cost, promoting positive values
    # for edges that can be added without exceeding capacity and are more beneficial
    # (lower cost in this case)
    heuristics = edge_feasibility_mask + cumulative_cost
    
    return heuristics