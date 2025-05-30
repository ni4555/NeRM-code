import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize distance matrix to make distances relative to the max distance in the matrix
    max_distance = torch.max(distance_matrix)
    normalized_distances = distance_matrix / max_distance

    # Calculate the cumulative demand along the diagonal, where each diagonal element represents
    # the total demand to visit the customer at that index
    cumulative_demand = demands

    # Combine normalized distance with cumulative demand to get heuristic values
    # This heuristic assumes that closer customers are more promising (negative value) and
    # that higher demands are less promising (negative value)
    heuristics = normalized_distances - cumulative_demand

    return heuristics