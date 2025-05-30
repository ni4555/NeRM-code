import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize them
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristic values based on the normalized demands
    # The heuristic function used here is a simple inverse of the demand
    # as a heuristic to prioritize edges with lower demand.
    # This is a simplistic heuristic and can be replaced with more sophisticated ones.
    heuristics = 1 / (normalized_demands + 1e-8)  # Adding a small constant to avoid division by zero
    
    return heuristics