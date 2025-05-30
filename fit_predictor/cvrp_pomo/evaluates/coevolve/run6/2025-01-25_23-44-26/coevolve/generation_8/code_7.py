import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic
    # Calculate inverse distances from the depot (node 0) to all other nodes
    inv_distance_heuristic = 1.0 / distance_matrix[0, 1:]

    # Load balancing heuristic
    # Calculate the load of each node if it was assigned to a vehicle
    load = torch.zeros_like(distance_matrix)
    load[0, 1:] = normalized_demands
    for i in range(1, len(demands)):
        load[i, 1:] = load[i-1, 1:] + normalized_demands[i]

    # Combine the heuristics
    combined_heuristic = inv_distance_heuristic - load

    return combined_heuristic