import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the potential of each edge
    potential_matrix = distance_matrix * normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)
    
    # Subtract the potential to make negative values for undesirable edges
    potential_matrix -= potential_matrix.max()
    
    return potential_matrix