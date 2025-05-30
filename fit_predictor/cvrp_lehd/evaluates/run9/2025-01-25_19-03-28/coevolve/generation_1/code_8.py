import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize customer demands by the vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the demand factor for each edge (i, j)
    demand_factor = torch.abs(normalized_demands) * torch.abs(distance_matrix)
    
    # Calculate the heuristic value for each edge
    heuristic_values = -demand_factor
    
    # Apply a threshold to promote certain edges over others
    threshold = torch.min(heuristic_values)
    heuristic_values[heuristic_values < threshold] += 1
    
    return heuristic_values