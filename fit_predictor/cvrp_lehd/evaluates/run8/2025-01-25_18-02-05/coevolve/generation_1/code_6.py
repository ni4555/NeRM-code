import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the difference in demand between each customer and the total capacity
    demand_diff = demands - demands.mean()
    
    # Calculate the sum of distances from each customer to all other customers
    distance_sum = torch.sum(distance_matrix, dim=1)
    
    # Use a simple heuristic that promotes edges with low demand difference and low total distance
    # The heuristic function is defined as the negative of the sum of demand difference and distance
    # to encourage selection of edges that contribute less to the total demand and distance
    heuristic_values = -torch.sum(torch.stack([demand_diff, distance_sum]), dim=1)
    
    return heuristic_values