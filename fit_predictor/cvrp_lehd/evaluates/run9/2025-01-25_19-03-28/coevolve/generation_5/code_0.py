import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand from the depot to each customer
    cumulative_demand = torch.cumsum(demands, dim=0)
    
    # Calculate the cumulative distance from the depot to each customer
    cumulative_distance = torch.cumsum(distance_matrix[0], dim=0)
    
    # Compute the heuristic value for each edge based on the cumulative demand and distance
    # We use the ratio of cumulative demand to cumulative distance as the heuristic value
    # This heuristic assumes that high demand edges are more promising
    heuristics = cumulative_demand / cumulative_distance
    
    # Normalize the heuristics to be in the range of [0, 1] and adjust to have positive values
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())
    heuristics = heuristics * 2 - 1  # Scaling to [-1, 1]
    
    return heuristics