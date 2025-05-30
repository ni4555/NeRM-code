import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to prevent very large distances from dominating the heuristic
    distance_matrix = torch.clamp(distance_matrix, min=1e-5)
    
    # Calculate the sum of demands to normalize them
    demand_sum = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / demand_sum
    
    # Calculate the heuristic values based on the distance and demand
    # Promising edges are those with lower distances and higher demand density
    # The heuristic is negative for undesirable edges and positive for promising ones
    heuristics = distance_matrix * normalized_demands
    
    return heuristics