import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand mask
    cumulative_demand = demands.cumsum(0)
    
    # Normalize the cumulative demand by the vehicle capacity
    normalized_demand = cumulative_demand / demands.sum()
    
    # Initialize the heuristics matrix with negative values
    heuristics = -torch.ones_like(distance_matrix)
    
    # Calculate the load difference for each edge
    load_difference = (normalized_demand.unsqueeze(1) - normalized_demand.unsqueeze(0))
    
    # Adjust heuristics based on load difference
    heuristics += load_difference * distance_matrix
    
    # Further refine heuristics by adding a term that penalizes larger distances
    heuristics += -distance_matrix * 0.1
    
    return heuristics