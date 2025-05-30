import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative weighted distance matrix
    # We use negative values to indicate shorter distances as more promising
    negative_weighted_distance = -distance_matrix

    # Integrate the normalized demand into the negative weighted distance
    # Higher demand should be more negative to indicate higher priority
    weighted_distance_matrix = negative_weighted_distance + normalized_demands.unsqueeze(1) * distance_matrix

    # Apply a threshold to filter out edges with very low weights (undesirable edges)
    # This threshold is a heuristic parameter that can be adjusted
    threshold = torch.min(weighted_distance_matrix) * 0.1
    promising_edges = weighted_distance_matrix > threshold

    # Convert boolean mask to a float tensor with the same values as the threshold
    # This will create negative values for undesirable edges and positive values for promising ones
    heuristics_matrix = torch.where(promising_edges, threshold, 0.0)

    return heuristics_matrix