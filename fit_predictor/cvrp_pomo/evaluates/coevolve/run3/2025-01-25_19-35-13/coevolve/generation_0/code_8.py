import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for each node (including the depot)
    total_demand = demands.sum()
    
    # Calculate the sum of demands for each edge (i.e., the demand if the edge is traversed)
    edge_demands = distance_matrix * demands
    
    # Calculate the difference between the total demand and the demand if the edge is traversed
    # This gives us a measure of how much the total demand would decrease if this edge is traversed
    demand_reduction = total_demand - edge_demands
    
    # To encourage visiting nodes that reduce the total demand, we use the negative of the demand reduction
    # This will result in positive values for edges that are promising and negative values for undesirable edges
    heuristics = -demand_reduction
    
    return heuristics