import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands to have a sum of 1
    demand_sum = demands.sum()
    normalized_demands = demands / demand_sum

    # Compute the inverse of the demands as a heuristic for the edges
    # Edges with lower demand (more capacity left) are considered more promising
    inverse_demands = 1 / normalized_demands

    # Compute the sum of the inverse demands for each edge (weighted by distance)
    # This gives a heuristic value for each edge based on the remaining capacity
    edge_heuristics = (inverse_demands * distance_matrix).sum(dim=1)

    # Return the heuristics matrix, with negative values for undesirable edges
    # (i.e., where the demand is already met or exceeded)
    return edge_heuristics - (inverse_demands * distance_matrix)