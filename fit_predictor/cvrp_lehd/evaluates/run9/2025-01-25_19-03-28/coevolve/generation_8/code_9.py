import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic value for each edge
    # Here we use a simple heuristic that considers the normalized demand and distance
    # A more complex heuristic could be designed here
    heuristics = normalized_demands * distance_matrix
    
    return heuristics