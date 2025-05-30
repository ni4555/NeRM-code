import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity by summing all customer demands
    total_capacity = demands.sum()
    
    # Normalize demands to the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the sum of distances for each edge in the matrix, scaled by demand
    edge_costs = distance_matrix * normalized_demands
    
    # Subtract the sum of distances from 1 to get heuristics (0-1 scale)
    heuristics = 1 - edge_costs
    
    # Ensure that the heuristics contain negative values for undesirable edges
    heuristics = torch.clamp(heuristics, min=0.0)
    
    return heuristics