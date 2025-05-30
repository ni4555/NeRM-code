import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize a tensor with zeros to hold the heuristic values
    heuristics = torch.zeros_like(distance_matrix)

    # Normalize demands to be within the range [0, 1]
    demands_normalized = demands / demands.sum()

    # Calculate the negative of the sum of demands for each edge
    negative_demand_sum = 1 - demands_normalized

    # Add the negative demand sum to the heuristic values for each edge
    heuristics += negative_demand_sum

    # Adjust the heuristic values based on the distance matrix
    # The idea here is to add a larger negative value for larger distances
    # which makes longer distances less promising.
    heuristics -= distance_matrix

    return heuristics