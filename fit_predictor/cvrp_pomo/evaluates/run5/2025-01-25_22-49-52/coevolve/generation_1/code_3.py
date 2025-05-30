import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands by total capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic for each edge
    # The heuristic is based on the demand of the destination node
    # Edges to nodes with higher normalized demand are considered more promising
    heuristics = distance_matrix * normalized_demands
    
    return heuristics