import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it's the sum of all demands)
    total_capacity = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_capacity
    
    # Calculate the potential for each edge based on distance and demand
    # We use a simple heuristic where we prefer edges with lower distance and higher demand
    potential = -distance_matrix + normalized_demands
    
    # Return the heuristics matrix
    return potential