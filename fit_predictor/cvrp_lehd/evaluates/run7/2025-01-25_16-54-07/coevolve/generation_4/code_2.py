import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot = 0  # Assuming the depot node is indexed by 0

    # Create a tensor to store the heuristics
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cumulative demand
    cumulative_demand = demands.cumsum(dim=0)

    # Initial routing with nearest neighbor
    for i in range(1, n):
        nearest = torch.argmin(distance_matrix[i, :i])  # Nearest node before i
        heuristics[i, nearest] = -float('inf')  # Mark the edge as undesirable
        heuristics[nearest, i] = -float('inf')  # Symmetrically mark the edge

    # Demand-driven optimization phase
    for i in range(1, n):
        if demands[i] > 0:
            # Find all feasible next nodes based on current vehicle capacity
            feasible_nodes = cumulative_demand[:i] <= demands[i]
            feasible_next_nodes = torch.nonzero(feasible_nodes, as_tuple=False).tolist()

            # For each feasible next node, set the edge as promising
            for node in feasible_next_nodes:
                heuristics[i, node] = 1.0

    return heuristics