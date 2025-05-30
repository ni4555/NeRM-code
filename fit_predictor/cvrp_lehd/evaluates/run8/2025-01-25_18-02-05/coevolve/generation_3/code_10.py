import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristics based on the normalized demands
    # We use a simple heuristic where the heuristic value is inversely proportional to the demand
    # and adjusted by the distance to the depot (which is 0 for the depot node).
    heuristics = -normalized_demands * distance_matrix

    # Set the heuristic value for the depot node to 0 since it is not an edge to be included in the solution
    heuristics[0] = 0

    return heuristics