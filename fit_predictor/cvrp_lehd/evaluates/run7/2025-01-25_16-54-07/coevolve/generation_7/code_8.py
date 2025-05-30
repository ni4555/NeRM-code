import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to be between 0 and 1
    demands_normalized = demands / demands.sum()
    
    # Calculate the cumulative demand mask
    cumulative_demand_mask = demands_normalized.cumsum(0)
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    # Using a simple heuristic where the heuristic is a combination of demand and distance
    # For example, a simple approach could be: heuristic = -distance + demand
    heuristic_matrix = -distance_matrix + cumulative_demand_mask
    
    # Ensure the heuristic matrix contains negative values for undesirable edges
    # and positive values for promising ones by clipping values
    heuristic_matrix = torch.clamp(heuristic_matrix, min=-1, max=0)
    
    return heuristic_matrix