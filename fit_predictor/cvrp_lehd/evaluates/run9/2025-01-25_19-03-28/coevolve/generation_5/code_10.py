import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized
    demands = demands / demands.sum()

    # Initialize heuristics matrix with the negative of the distances
    heuristics = -distance_matrix

    # Incorporate customer demand into the heuristics
    # The idea here is that the more the demand of the customer, the less attractive it is to include that edge
    heuristics += demands * distance_matrix

    # Normalize heuristics to ensure no negative values (use min to prevent underflow)
    min_val = heuristics.min()
    heuristics += min_val  # This ensures the heuristics are all non-negative
    heuristics /= heuristics.max()  # Normalize the values to make them comparable

    return heuristics