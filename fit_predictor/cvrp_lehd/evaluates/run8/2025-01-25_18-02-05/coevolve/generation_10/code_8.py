import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for the depot (sum of all demands should equal 1)
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Compute the heuristic values based on the ratio of customer demand to distance
    # This heuristic assumes that closer customers with higher demands are more promising
    heuristic_matrix = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)
    
    # Subtract the sum of the heuristic values from the distance to penalize longer distances
    # This is a common heuristic called the "sum of squares" heuristic
    heuristic_matrix -= distance_matrix.sum(dim=1, keepdim=True)
    heuristic_matrix -= distance_matrix.sum(dim=0, keepdim=True)
    
    # Add a small constant to avoid division by zero and ensure non-negative values
    epsilon = 1e-6
    heuristic_matrix += epsilon
    
    # Normalize the heuristic matrix to have non-negative values and to maintain the relative importance
    heuristic_matrix = (heuristic_matrix - heuristic_matrix.min()) / (heuristic_matrix.max() - heuristic_matrix.min())
    
    return heuristic_matrix