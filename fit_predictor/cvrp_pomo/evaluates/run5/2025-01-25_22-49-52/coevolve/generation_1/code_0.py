import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    # Calculate the normalized demand difference for each edge
    demand_diff = demands.unsqueeze(1) - demands.unsqueeze(0)
    # Calculate the heuristic values based on the demand differences
    # Using a simple heuristic that penalizes large demand differences
    heuristic_values = -torch.abs(demand_diff)
    # Normalize the heuristic values to have a range of [0, 1]
    max_val = torch.max(heuristic_values)
    min_val = torch.min(heuristic_values)
    heuristic_values = (heuristic_values - min_val) / (max_val - min_val)
    # Adjust the values to be positive
    heuristic_values = heuristic_values * (1 - heuristic_values)
    return heuristic_values