import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming demands are normalized, we can use a simple heuristic based on the demand
    # For example, we could use the negative demand for each edge as a heuristic.
    # This assumes that we want to prioritize edges with lower demand (which would
    # encourage the formation of routes with more stops).
    # We use negative demand to ensure that PyTorch's sorting operations can be used
    # to easily select the best edges.
    negative_demands = -demands
    # Use broadcasting to create a matrix where each cell is the negative demand from the depot to the customer
    demand_matrix = negative_demands.view(-1, 1) + negative_demands.view(1, -1)
    # The distance_matrix is already a 2D tensor of distances, so we can directly subtract it from the demand_matrix
    distance_subtracted = demand_matrix - distance_matrix
    return distance_subtracted