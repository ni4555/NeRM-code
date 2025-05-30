import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the total distance from the depot to all other nodes
    depot_distances = distance_matrix[:, 0]
    # Calculate the total demand from the depot to all other nodes
    total_demand = demands[1:].sum()
    # Calculate the heuristic value for each edge
    heuristics = -depot_distances * demands[1:]
    # Normalize the heuristic values by the total demand
    heuristics /= total_demand
    # Clip the values to ensure they are within a certain range
    heuristics = torch.clamp(heuristics, min=-1.0, max=1.0)
    return heuristics