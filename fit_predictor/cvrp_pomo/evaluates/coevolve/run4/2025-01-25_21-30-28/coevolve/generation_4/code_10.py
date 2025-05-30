import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the load factor for each customer
    load_factors = demands / demands.sum()
    
    # Calculate the heuristics based on distance and load factor
    # Negative values for undesirable edges, positive values for promising ones
    heuristics = distance_matrix * load_factors
    
    return heuristics