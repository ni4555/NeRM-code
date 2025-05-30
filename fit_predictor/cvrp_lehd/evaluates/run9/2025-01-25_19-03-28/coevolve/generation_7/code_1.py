import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity (assuming it's a single value for all vehicles)
    total_capacity = demands.sum()
    
    # Calculate the difference in demands from the normalized demands
    # This will be used to penalize unbalanced loads
    load_difference = demands - demands.mean()
    
    # Calculate the heuristic value as a combination of distance, demand difference,
    # and a normalization factor to ensure all values are on the same scale
    # Here we use a simple normalization where we divide by the total capacity
    # and add a small positive constant to avoid division by zero.
    heuristic_value = (distance_matrix + load_difference ** 2).div(total_capacity + 1e-10)
    
    # We want to encourage short distances and balanced loads, hence we use negative values
    # for undesirable edges and positive values for promising ones.
    # We subtract the heuristic value from a large number to get negative values.
    # The large number should be larger than the maximum possible heuristic value.
    max_heuristic = torch.max(heuristic_value)
    promising_edges = (heuristic_value - max_heuristic).neg()
    
    return promising_edges