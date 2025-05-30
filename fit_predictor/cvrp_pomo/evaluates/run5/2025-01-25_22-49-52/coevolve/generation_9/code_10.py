import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = torch.sum(demands)
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values based on normalized demands
    # Here we use a simple heuristic: the inverse of the demand multiplied by the distance
    # This heuristic assumes that nodes with lower demand are more promising
    heuristics = 1 / (normalized_demands * distance_matrix)

    # Replace NaNs with a very small number to avoid division by zero
    heuristics = torch.nan_to_num(heuristics, nan=1e-10)

    # Ensure that all values are non-negative
    heuristics = torch.clamp(heuristics, min=0)

    return heuristics