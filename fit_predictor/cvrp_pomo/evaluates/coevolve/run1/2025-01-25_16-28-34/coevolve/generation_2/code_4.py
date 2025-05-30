import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demand for each customer
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cost of each edge as the negative demand times the distance
    # This heuristic is based on the fact that edges with lower demand and distance are more promising
    heuristics = -torch.mul(normalized_demands, distance_matrix)

    # We can add more complexity here, for instance:
    # 1. Apply a trade-off factor to balance between distance and demand
    # 2. Incorporate additional heuristics like min-cost max-flow

    return heuristics