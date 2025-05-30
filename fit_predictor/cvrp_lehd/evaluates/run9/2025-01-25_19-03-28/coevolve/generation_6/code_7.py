import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the sum of normalized demands for each edge
    edge_normalized_demand_sum = (distance_matrix * normalized_demands.unsqueeze(1)).sum(0)
    
    # Calculate the heuristics values
    heuristics = distance_matrix - edge_normalized_demand_sum
    
    # Ensure the heuristics contain negative values for undesirable edges
    heuristics = heuristics.clamp(min=-1e-6)
    
    return heuristics