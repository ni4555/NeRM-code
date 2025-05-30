import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total demand to normalize the demands
    total_demand = demands.sum()
    
    # Normalize the demands
    normalized_demands = demands / total_demand
    
    # Calculate the potential heuristics
    # We use a simple heuristic where we consider the demand of the node
    # multiplied by the inverse of the distance to the depot as a heuristic value
    # This heuristic is inspired by the EUCALC heuristic for the TSP
    heuristics = demands * (1.0 / distance_matrix)
    
    # Normalize the heuristics to ensure they are within the same scale as the demands
    heuristics /= total_demand
    
    return heuristics