import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values based on normalized demands
    # Negative values for undesirable edges (e.g., higher distances)
    # Positive values for promising edges (e.g., lower distances)
    heuristics = -distance_matrix + normalized_demands.unsqueeze(1) * distance_matrix
    
    return heuristics