import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the total demand weight for each edge
    # For an edge from node i to node j, the weight is the sum of demands of customers visiting node j
    edge_demand_weights = torch.zeros_like(distance_matrix)
    for i in range(1, distance_matrix.shape[0]):
        edge_demand_weights[:, i] = torch.cumsum(normalized_demands[i:], dim=0)[:-1]

    # Calculate the heuristic values based on the normalized demands and the sum of edge weights
    heuristics = -edge_demand_weights - distance_matrix

    # To ensure that some edges are not included (non-promising edges), add a small constant to the distance matrix
    # The larger the distance, the less promising the edge is
    heuristics += torch.sum(distance_matrix, dim=1).unsqueeze(1)

    return heuristics