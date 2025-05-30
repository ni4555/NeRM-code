import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming distance_matrix and demands are already normalized and properly shaped.
    
    # Calculate the sum of demands excluding the depot node
    total_demand = demands.sum()
    
    # Calculate the maximum possible demand for any vehicle
    max_demand = demands.max()
    
    # Calculate the remaining capacity after assigning the maximum demand
    remaining_capacity = total_demand - max_demand
    
    # Calculate the heuristics: higher negative values for edges with higher demands,
    # and add a penalty for edges that are not part of the maximum demand.
    heuristics = -demands + remaining_capacity
    
    return heuristics