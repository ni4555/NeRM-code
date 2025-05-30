import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()
    
    # Normalize the demands vector to represent the fraction of the total capacity
    normalized_demands = demands / total_capacity
    
    # Compute the heuristics using the normalized demands
    heuristics = distance_matrix * normalized_demands
    
    # Ensure that the heuristics contain negative values for undesirable edges and
    # positive values for promising ones by adding a large positive value to the
    # negative heuristics and subtracting it from the positive heuristics
    max_value = heuristics.max()
    min_value = heuristics.min()
    heuristics = (heuristics - min_value) * 2 / (max_value - min_value) - 1
    
    return heuristics