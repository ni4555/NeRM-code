import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the cumulative demand along the main diagonal (the diagonal represents the distance from the depot to itself,
    # which is 0 and doesn't contribute to the heuristic, hence we skip it in the sum)
    cum_demand = demands.cumsum(dim=0)
    
    # Subtract the demands from the cumulative demand matrix to get the difference between consecutive demands
    diff_demand = cum_demand[:-1] - cum_demand[1:]
    
    # Compute the heuristic values using a simple heuristic where the difference in demand between consecutive nodes
    # is used to estimate the edge utility. We can scale this by some factor (e.g., 1/distance) or make it negative for longer
    # edges or higher demand differences.
    # In this simple version, we will use the difference in demand as is.
    heuristic_matrix = -diff_demand
    
    return heuristic_matrix