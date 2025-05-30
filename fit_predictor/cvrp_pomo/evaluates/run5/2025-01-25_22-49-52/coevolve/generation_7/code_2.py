import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Normalize the distance matrix by dividing by the maximum distance
    max_distance = distance_matrix.max()
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the potential value for each edge
    # The potential value is a function of both normalized demand and normalized distance
    potential_value = normalized_distance_matrix * normalized_demands

    # We want to steer the GA to desirable edges, so we take the negative of the potential value
    # to ensure that positive values indicate promising edges
    heuristics = -potential_value

    return heuristics