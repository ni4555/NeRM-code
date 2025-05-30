import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of all demands
    total_demand = demands.sum()
    
    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_demand
    
    # Compute the heuristics values based on the normalized demands
    heuristics = distance_matrix * normalized_demands
    
    # Introduce a small constant to avoid division by zero in the next step
    epsilon = 1e-10
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min() + epsilon)
    
    # Invert the heuristics to have negative values for undesirable edges and positive for promising ones
    heuristics = 1 - heuristics
    
    return heuristics