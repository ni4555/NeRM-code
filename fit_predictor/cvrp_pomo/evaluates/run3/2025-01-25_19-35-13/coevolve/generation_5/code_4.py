import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize the demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the heuristic values
    # The heuristic is a combination of demand and distance, weighted
    # For example, we could use the demand as a negative value and distance as a positive value
    # Here we use a simple heuristic: demand * distance
    # Negative demand values encourage the inclusion of edges with lower demand
    # Positive distance values encourage the inclusion of edges with shorter distance
    # The weights can be adjusted to favor one over the other
    demand_weight = -1.0
    distance_weight = 1.0
    heuristic_matrix = demand_weight * normalized_demands.unsqueeze(1) * distance_matrix + \
                       distance_weight * normalized_demands.unsqueeze(0) * distance_matrix

    # To avoid zero heuristic values (which might not be desirable for certain metaheuristics),
    # we add a small constant to the diagonal, but only for the edges that are part of the route
    # We do this by creating a matrix of ones and subtracting the identity matrix
    route_matrix = (torch.eye(n) - torch.eye(n, dtype=torch.bool)).bool()
    heuristic_matrix[route_matrix] += 1e-5

    return heuristic_matrix