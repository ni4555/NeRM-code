import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of demands
    total_demand = demands.sum()
    
    # Normalize the demands by the total vehicle capacity
    normalized_demands = demands / total_demand
    
    # Compute the heuristic values as negative of the normalized demand
    # The idea is that lower demand implies higher priority for that edge
    heuristics = -normalized_demands
    
    # Adjust the heuristic values to ensure there are positive values for promising edges
    # This could be a thresholding operation, or a more sophisticated scaling
    # For simplicity, we will use a threshold
    threshold = torch.min(heuristics) + 1
    heuristics = torch.where(heuristics < threshold, threshold, heuristics)
    
    return heuristics