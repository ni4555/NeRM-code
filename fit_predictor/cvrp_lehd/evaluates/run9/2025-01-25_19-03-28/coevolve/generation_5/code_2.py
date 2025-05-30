import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the input matrices have the same size
    if distance_matrix.shape != demands.shape:
        raise ValueError("The size of the distance matrix and demands vector must match.")
    
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity
    
    # Calculate the inverse of demand (heuristic value) and add it to the distance matrix
    # This assumes that the lower the demand, the higher the heuristic value for an edge
    inverse_demand = 1 / (normalized_demands + 1e-10)  # Adding a small epsilon to avoid division by zero
    heuristics = distance_matrix + inverse_demand
    
    return heuristics