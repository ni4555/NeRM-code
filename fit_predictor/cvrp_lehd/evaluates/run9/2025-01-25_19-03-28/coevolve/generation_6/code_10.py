import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the negative of the distance matrix for the heuristic approach
    negative_distance_matrix = -distance_matrix

    # Calculate the total vehicle capacity as a scalar tensor
    total_capacity = demands.sum()

    # Normalize customer demands by the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Calculate the potential cost of each edge as a product of distance and demand
    potential_cost = negative_distance_matrix * normalized_demands

    # Apply a threshold to transform potential cost into heuristic values:
    # Positive for promising edges and negative for undesirable edges
    # Here, 0 is chosen as a threshold value; you may want to adjust this based on your specific problem context
    threshold = 0
    heuristics = torch.where(potential_cost > threshold, potential_cost, -torch.inf)

    return heuristics