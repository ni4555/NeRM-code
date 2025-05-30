import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)

    # Calculate the total vehicle capacity by summing all demands
    total_capacity = demands.sum()

    # Calculate the normalized demands (demands divided by total capacity)
    normalized_demands = demands / total_capacity

    # Compute the potential benefit of each edge
    # This heuristic considers the difference in normalized demand from the source node ( depot )
    # The larger the difference, the more promising the edge is to include in the solution
    edge_benefit = (distance_matrix - normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)).abs()

    # To make the heuristic more promising for edges with high demand differences, we can add a bonus
    # This bonus can be a function of the total capacity to ensure the heuristic encourages load balancing
    bonus = torch.clamp(1 - normalized_demands, min=0) * total_capacity

    # Combine the edge benefit with the bonus to get the final heuristic values
    heuristics = edge_benefit + bonus

    # To ensure the heuristic values are in a range that is suitable for optimization algorithms
    # We normalize the heuristic values by subtracting the minimum value and then dividing by the maximum value
    heuristics -= heuristics.min()
    heuristics /= heuristics.max()

    return heuristics