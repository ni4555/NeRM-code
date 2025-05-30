import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands to the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the cost for each edge
    edge_costs = torch.abs(distance_matrix - distance_matrix.transpose(0, 1))
    
    # Calculate the potential overload cost
    potential_overload = torch.clamp(normalized_demands - 1, min=0)
    
    # Calculate the heuristics values
    heuristics = -edge_costs + potential_overload
    
    # Normalize the heuristics matrix to ensure all values are in the same scale
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    return heuristics