import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot = 0
    # Calculate the distance from the depot to all other nodes
    depot_distances = distance_matrix[depot, :]
    # Calculate the total demand
    total_demand = demands.sum()
    # Normalize demands by total vehicle capacity
    normalized_demands = demands / total_demand
    # Calculate the heuristics value for each edge
    heuristics = -depot_distances * normalized_demands
    return heuristics