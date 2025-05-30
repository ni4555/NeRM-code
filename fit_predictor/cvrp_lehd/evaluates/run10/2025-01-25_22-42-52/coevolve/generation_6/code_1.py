import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Create a vector of all ones, which will be used to calculate the total demand for each edge
    ones = torch.ones_like(distance_matrix, dtype=torch.float32)

    # Calculate the total demand for each edge by multiplying the demand at the source node by 1s
    total_demand_per_edge = torch.dot(normalized_demands, ones)

    # Compute the heuristic values by subtracting the total demand per edge from the distance matrix
    # This encourages including edges with lower demand first, which can be a good heuristic
    heuristics = distance_matrix - total_demand_per_edge

    # Set negative values to a small negative value to indicate undesirable edges
    # The choice of small negative value depends on the scale of the distance matrix
    undesirable_threshold = -0.01
    heuristics[heuristics < undesirable_threshold] = undesirable_threshold

    return heuristics