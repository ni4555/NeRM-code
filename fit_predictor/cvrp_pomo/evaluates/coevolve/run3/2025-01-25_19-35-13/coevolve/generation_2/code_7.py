import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming the depot is node 0 and its demand is 0, so we can remove the first row and column
    # from distance_matrix and demands, as they will not contribute to the heuristics.
    distance_matrix = distance_matrix[1:, 1:]
    demands = demands[1:]

    # Normalize demands by the total capacity to ensure that the sum of demands on any route
    # does not exceed the vehicle capacity.
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the potential heuristics based on normalized demands.
    # We use negative values for undesirable edges and positive values for promising ones.
    heuristics = -distance_matrix + normalized_demands

    # To encourage visiting customers before moving back to the depot, add a small positive value
    # for edges that go from a customer to the depot.
    heuristics[:, 0] += 1

    return heuristics