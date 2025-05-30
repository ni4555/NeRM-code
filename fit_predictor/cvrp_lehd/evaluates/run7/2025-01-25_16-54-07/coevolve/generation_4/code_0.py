import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands vector with respect to the vehicle capacity (for this example, let's assume capacity is 1)
    normalized_demands = demands / demands.sum()

    # Initialize a mask with negative values, which will represent undesirable edges
    edge_mask = -torch.ones_like(distance_matrix)

    # Define the nearest neighbor heuristic: for each node, find the nearest node
    for i in range(1, distance_matrix.size(0)):
        min_distance, min_index = torch.min(distance_matrix[i], dim=0)
        edge_mask[i, min_index] = 1

    # Incorporate demand checks
    for i in range(1, distance_matrix.size(0)):
        for j in range(1, distance_matrix.size(0)):
            cumulative_demand = (distance_matrix[i, j] * (1 - normalized_demands[i]) +
                                 distance_matrix[i, j] * (normalized_demands[j] - normalized_demands[i]))
            if cumulative_demand <= 1:  # If the edge does not cause the vehicle to be overloaded
                edge_mask[i, j] = min(edge_mask[i, j], cumulative_demand)

    # Normalize the heuristics to be within the range (-1, 1)
    edge_mask = torch.sigmoid(edge_mask)
    edge_mask = (2 * edge_mask - 1)  # Scaling to range [-1, 1]

    return edge_mask