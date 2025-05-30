import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the potential heuristics for each edge
    # Here we use a simple heuristic that considers the demand and distance
    # Negative values are used for undesirable edges, positive for promising ones
    heuristics = -distance_matrix * normalized_demands
    
    # To make the heuristic more meaningful, we can add a term that encourages
    # visiting customers with higher demand first, but we should not exceed the capacity
    # For simplicity, we'll just add a small positive value for high-demand edges
    high_demand_bonus = torch.where(normalized_demands > 0.5, 1.0, 0.0)
    heuristics += high_demand_bonus
    
    return heuristics