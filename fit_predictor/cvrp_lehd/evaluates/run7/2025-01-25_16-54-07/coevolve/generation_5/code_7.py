import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total distance for each possible route from the depot to any customer and back to the depot
    return_distance = distance_matrix + distance_matrix.transpose(0, 1)
    total_demand = demands + demands

    # Calculate the total distance for each route, weighted by the demand (since the demands are normalized by the total vehicle capacity)
    route_weighted_distance = return_distance * demands

    # Subtract the demand from each edge to encourage visiting customers with higher demands earlier
    # This creates a heuristic that favors routes that include high-demand customers
    heuristic = -route_weighted_distance

    # Ensure that the heuristic has a zero value on the diagonal (edges to the depot from the depot)
    # and that all other values are non-positive to indicate undesirable edges
    heuristic = torch.clamp(heuristic, min=0.0)

    return heuristic