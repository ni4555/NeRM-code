import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands for normalization
    total_demand = demands.sum()
    
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Calculate the heuristics as the negative of the distances
    # multiplied by the normalized demands, since we want to favor
    # shorter distances with higher demands
    heuristics = -distance_matrix * normalized_demands
    
    # Clip the values to ensure that they are within a certain range
    # to avoid numerical issues and to ensure non-negative values
    heuristics = torch.clamp(heuristics, min=-1e-6, max=1e-6)
    
    return heuristics