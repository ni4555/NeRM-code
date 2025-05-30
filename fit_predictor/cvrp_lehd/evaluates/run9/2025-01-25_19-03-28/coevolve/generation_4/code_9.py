import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize the demand vector by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics values
    # Use a simple heuristic that considers the demand ratio to emphasize high-demand edges
    # Here we also introduce a decay factor for distance to make longer distances less attractive
    decay_factor = 0.1
    distance_decay = torch.exp(-decay_factor * distance_matrix)
    demand_factor = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Combine factors to get heuristics
    heuristics = (distance_decay * demand_factor).sum(dim=2)
    
    return heuristics