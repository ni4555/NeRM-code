import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # Here we use a simple heuristic based on the normalized demand
    # and the distance. A negative heuristic is assigned to edges
    # that should not be included in the solution.
    heuristic_matrix = -distance_matrix * normalized_demands
    
    return heuristic_matrix