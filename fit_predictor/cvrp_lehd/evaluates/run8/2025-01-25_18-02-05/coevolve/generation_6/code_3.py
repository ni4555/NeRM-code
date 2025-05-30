import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands to be the fraction of the total capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the heuristics values as negative of the distance times the normalized demand
    # This encourages paths that visit nodes with higher demand (normalized demand)
    heuristics_values = -distance_matrix * normalized_demands
    
    # Clip the values to ensure that we have negative values for undesirable edges
    # and positive values for promising ones, avoiding any zero or negative values
    heuristics_values = torch.clamp(heuristics_values, min=-1e-6)
    
    return heuristics_values