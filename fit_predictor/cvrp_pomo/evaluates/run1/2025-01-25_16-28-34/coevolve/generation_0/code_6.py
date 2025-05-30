import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands tensor includes the depot demand, which is zero
    if demands[0] != 0:
        raise ValueError("Demands vector must have the depot demand at the first index (0) and be zero.")

    # Compute the heuristic as a function of the distance and the normalized demand
    # Negative values for the heuristic mean "undesirable" edges, while positive values are "promising".
    # Here we are using the reciprocal of the distance (which is smaller for shorter distances) and
    # subtracting the normalized demand to penalize heavier demands.
    heuristic_matrix = 1.0 / (distance_matrix + 1e-10) - demands[1:]  # Add a small epsilon to avoid division by zero

    return heuristic_matrix