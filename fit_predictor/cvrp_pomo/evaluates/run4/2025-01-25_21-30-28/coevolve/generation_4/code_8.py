import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity as the sum of demands
    total_capacity = demands.sum()
    
    # Normalize demands to represent the fraction of vehicle capacity each customer demand is
    normalized_demands = demands / total_capacity
    
    # Compute the sum of normalized demands for each node
    node_demand_sums = normalized_demands.sum(dim=1)
    
    # Calculate the heuristic value for each edge
    # The heuristic is a combination of the negative distance and the load at the destination node
    heuristics = -distance_matrix + node_demand_sums
    
    return heuristics