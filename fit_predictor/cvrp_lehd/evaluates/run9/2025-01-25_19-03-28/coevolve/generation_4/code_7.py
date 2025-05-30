import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    demands = demands / demands.sum()

    # Compute the heuristic values for each edge
    # For this simple heuristic, we consider the inverse of the demand as a measure of
    # promise, meaning high demand will contribute negatively to the heuristic (undesirable).
    # This is just an example, real heuristics might be more complex.
    heuristic_values = -1 * (distance_matrix * demands.unsqueeze(1)).sum(2)

    # The resulting matrix should be of the same shape as the distance matrix.
    assert heuristic_values.shape == distance_matrix.shape

    return heuristic_values