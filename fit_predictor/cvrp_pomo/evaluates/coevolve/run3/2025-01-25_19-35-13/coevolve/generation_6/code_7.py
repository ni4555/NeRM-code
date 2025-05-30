import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    vehicle_capacity = demands.sum()
    
    # Normalize demands by vehicle capacity
    normalized_demands = demands / vehicle_capacity
    
    # Calculate the sum of demands for each edge
    edge_demands = (distance_matrix * normalized_demands.unsqueeze(1) +
                    distance_matrix.unsqueeze(0) * normalized_demands)
    
    # Use the sum of demands as the heuristic value for each edge
    heuristics = edge_demands.sum(dim=1)
    
    return heuristics