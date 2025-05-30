import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the demands vector is already normalized by the total vehicle capacity
    # We calculate the maximum demand for normalization purposes
    max_demand = demands.max()
    
    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)
    
    # Calculate the Euclidean distance heuristic
    heuristics += distance_matrix
    
    # Calculate the demand-based heuristic (promising edges have lower demand)
    heuristics -= demands
    
    # Apply a normalization to ensure non-negative values
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    
    # Cap the heuristics at a maximum positive value to avoid negative values
    heuristics = torch.clamp(heuristics, min=0)
    
    return heuristics