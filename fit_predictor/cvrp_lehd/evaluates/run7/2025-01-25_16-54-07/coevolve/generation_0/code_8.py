import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics matrix
    # For each edge, the heuristic is the demand at the destination node
    # multiplied by the inverse of the demand at the source node
    heuristics_matrix = normalized_demands * (1 / normalized_demands)
    
    # Replace negative values with a very small negative value to mark undesirable edges
    heuristics_matrix[heuristics_matrix < 0] = -1e10
    
    return heuristics_matrix