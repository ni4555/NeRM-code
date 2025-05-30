import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are tensors
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    demands = torch.tensor(demands, dtype=torch.float32)
    
    # Normalize demands by the total capacity for comparison purposes
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the heuristic values for each edge
    # We use a simple heuristic that combines distance and normalized demand
    # Negative values are undesirable edges, positive values are promising ones
    # We subtract the normalized demand from the distance to give a priority to edges
    # with lower demand and lower distance
    heuristics = distance_matrix - normalized_demands
    
    return heuristics