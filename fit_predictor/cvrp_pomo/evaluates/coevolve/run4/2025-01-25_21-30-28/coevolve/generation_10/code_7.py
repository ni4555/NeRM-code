import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the inverse of the demands, which are normalized by total vehicle capacity
    inverse_demands = 1.0 / (demands - demands.min())

    # Compute the inverse distance heuristic: lower distances should be more promising
    inverse_distance = 1.0 / distance_matrix

    # Compute the Normalization heuristic: normalize distances by demands
    # This step gives higher priority to closer customers that are under-demand
    normalization = distance_matrix * inverse_demands

    # The Normalization heuristic gives a negative score for under-demand and positive for over-demand
    # We convert this to a promising score (negative for undesirable, positive for promising)
    # by negating the Normalization heuristic values
    normalization = -normalization

    # Combine the heuristics using a weighted sum
    # You can adjust the weights (alpha and beta) as needed for the specific problem instance
    alpha = 0.5
    beta = 0.5
    combined_heuristic = alpha * inverse_distance + beta * normalization

    return combined_heuristic