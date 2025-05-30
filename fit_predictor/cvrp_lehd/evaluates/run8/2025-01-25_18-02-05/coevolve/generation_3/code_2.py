import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands to normalize
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as a product of distances and normalized demands
    # This heuristic is a simple inverse demand heuristic
    heuristics = distance_matrix * (1 / (normalized_demands + 1e-6))  # Adding a small constant to avoid division by zero
    
    return heuristics