import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Calculate the relative demand for each customer
    relative_demands = demands / total_demand
    
    # Compute the heuristics based on the relative demand and distance
    # Promising edges will have a positive score and undesirable edges will have a negative score
    heuristics = relative_demands * distance_matrix
    
    return heuristics