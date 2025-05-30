import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix
    max_distance = torch.max(distance_matrix)
    distance_matrix = distance_matrix / max_distance

    # Normalize demands by the total vehicle capacity (assuming total capacity is 1 for simplicity)
    total_demand = torch.sum(demands)
    demands = demands / total_demand

    # Calculate potential values for explicit depot handling
    # We will use the inverse of demands as a proxy for this
    depot_potential = 1 / demands

    # Calculate edge potential values
    # This is a simple heuristic that encourages edges that are close to the depot and have low demand
    edge_potential = distance_matrix * depot_potential

    # Incorporate constraint programming by ensuring that the sum of edge potentials does not exceed the vehicle capacity
    # This is a simple approach, as we are not actually enforcing the capacity constraint here, but rather using it to guide the heuristic
    # We will use a threshold that is a bit less than 1 to ensure the capacity is not exactly met
    capacity_threshold = 0.95

    # Apply a dynamic window approach to adapt to dynamic changes in problem instances
    # For simplicity, we will just subtract a small value from the edge potentials that are too promising
    edge_potential = torch.where(edge_potential > capacity_threshold, edge_potential - 0.1, edge_potential)

    # Ensure that all edges from the depot are given a positive potential value
    # This is to encourage the algorithm to consider visiting the depot
    edge_potential = torch.where(torch.eq(distance_matrix.sum(dim=1), 1), edge_potential + 1e5, edge_potential)

    return edge_potential