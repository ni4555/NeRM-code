import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the demands vector is broadcastable to the shape of distance_matrix
    demands = demands.view(-1, 1)
    
    # Calculate the cumulative demand along each row (from the start node)
    cumulative_demand = torch.cumsum(demands, dim=1)
    
    # Calculate the cumulative demand along each column (to the start node)
    cumulative_demand_t = torch.cumsum(demands, dim=0).transpose(0, 1)
    
    # Calculate the remaining capacity for each vehicle along each edge
    remaining_capacity = (1 - cumulative_demand) * (1 - cumulative_demand_t)
    
    # Assign a heuristic value to each edge based on the remaining capacity
    # We use a positive heuristic for promising edges and negative for undesirable ones
    # Promising edges have high remaining capacity, undesirable edges have low or negative remaining capacity
    heuristics = remaining_capacity - 1  # Shift the scale to be positive for promising edges
    
    # Handle the diagonal elements (edges to and from the start node) by setting them to a very low value
    # This prevents the start node from being included in the solution
    heuristics.diag().fill_(float('-inf'))
    
    return heuristics