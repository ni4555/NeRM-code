import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the inputs are valid
    if distance_matrix.ndim != 2 or demands.ndim != 1 or distance_matrix.shape[0] != distance_matrix.shape[1] or demands.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Invalid input shapes")

    # Normalize demands to be between 0 and 1
    total_demand = demands.sum()
    demands = demands / total_demand

    # Calculate the negative cost heuristics based on demand and distance
    # For edges with demand, we assign a positive heuristic (inverse demand)
    # For edges without demand (to depot or between customers with 0 demand), assign a negative heuristic (distance)
    # Note: We avoid the diagonal and the edge from the last customer to the depot since they have 0 demand
    edge_demand = demands.unsqueeze(1) * demands.unsqueeze(0)  # Demand for each edge
    negative_distance = -distance_matrix
    heuristics = negative_distance + edge_demand

    # Set heuristics to zero where there is no demand to prevent selecting these edges
    heuristics[torch.triu_indices(heuristics.shape[0], heuristics.shape[1], k=1)] = 0
    heuristics[torch.tril_indices(heuristics.shape[0], heuristics.shape[1], k=-1)] = 0
    heuristics[torch.tensor([0], dtype=torch.long)] = 0  # The depot edge to itself has no demand

    return heuristics