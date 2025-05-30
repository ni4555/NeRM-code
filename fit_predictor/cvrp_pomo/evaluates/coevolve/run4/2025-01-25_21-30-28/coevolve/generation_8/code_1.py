import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse distance matrix
    inverse_distance_matrix = 1.0 / distance_matrix

    # Normalize demands by the total vehicle capacity
    normalized_demands = demands / demands.sum()

    # Combine inverse distance and normalized demand to get the heuristic values
    # Note: The sum of all heuristic values should be equal to the sum of all demands
    # as we normalize the demands to be a probability distribution.
    heuristics = -inverse_distance_matrix * normalized_demands

    # Clip the values to ensure that they are within a reasonable range
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)

    return heuristics