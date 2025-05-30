import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demands from each customer to the depot
    demand_diff = demands - demands[0]
    
    # Calculate the sum of absolute demand differences for each edge
    edge_demand_sum = torch.abs(demand_diff)
    
    # Calculate the maximum absolute demand difference for each edge
    max_demand_diff = torch.max(edge_demand_sum, dim=0).values
    
    # Calculate the negative of the distance matrix to use for minimizing distance
    negative_distance = -distance_matrix
    
    # Calculate the heuristic values by combining the maximum demand difference and negative distance
    heuristics = max_demand_diff + negative_distance
    
    return heuristics