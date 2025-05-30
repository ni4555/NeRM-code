import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_vector = demands / total_capacity

    # Calculate the initial heuristic values based on the inverse of demand
    initial_heuristics = 1 / demand_vector

    # Apply a penalty for edges that are too long (arbitrary threshold, could be tuned)
    long_edge_penalty = 0.1
    edge_length_threshold = 1.5
    long_edge_mask = distance_matrix > edge_length_threshold
    initial_heuristics[long_edge_mask] -= long_edge_penalty

    # Normalize the heuristics to ensure they sum to the total number of edges
    heuristic_sum = initial_heuristics.sum()
    heuristics = initial_heuristics / heuristic_sum

    return heuristics