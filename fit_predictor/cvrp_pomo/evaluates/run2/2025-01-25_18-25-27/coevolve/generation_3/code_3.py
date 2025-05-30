import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the absolute difference between demands to identify potential hotspots
    demand_diff = torch.abs(demands - demands.mean())
    
    # Calculate the sum of demands up to each node to identify potential load peaks
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Use the distance matrix to calculate the distance from the depot to each node
    distance_from_depot = distance_matrix[:, 0]
    
    # Combine the factors to get heuristic values
    heuristic_values = -demand_diff * cumulative_demand * distance_from_depot
    
    return heuristic_values