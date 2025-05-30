import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to sum to 1 for easier interpretation
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum
    
    # Calculate the total demand for each edge
    total_demand = torch.matmul(normalized_demands, distance_matrix)
    
    # Calculate the heuristic value for each edge
    # The heuristic is negative for undesirable edges (high demand) and positive for promising ones (low demand)
    heuristics = -total_demand
    
    return heuristics