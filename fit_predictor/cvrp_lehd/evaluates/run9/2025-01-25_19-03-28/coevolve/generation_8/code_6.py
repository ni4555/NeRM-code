import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize
    total_demand = demands.sum()
    
    # Normalize demands
    normalized_demands = demands / total_demand
    
    # Calculate the load factor for each customer
    load_factor = normalized_demands * distance_matrix
    
    # Use the load factor as a heuristic to prioritize edges with lower load factor
    heuristics = load_factor.sum(dim=1) - load_factor.sum(dim=0)
    
    return heuristics