import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands have the correct shape
    assert distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]
    assert demands.ndim == 1 and demands.shape[0] == distance_matrix.shape[0]

    # Normalize demands to have a sum of 1 (for example, for the sum-to-one normalization)
    normalized_demands = demands / demands.sum()

    # Initialize a tensor of zeros with the same shape as the distance matrix
    heuristics = torch.zeros_like(distance_matrix)

    # Compute the heuristics: we can use a simple heuristic here, such as the inverse of the demand
    # multiplied by the distance, which encourages selecting edges with lower demand and shorter distance
    heuristics = -distance_matrix * normalized_demands

    return heuristics