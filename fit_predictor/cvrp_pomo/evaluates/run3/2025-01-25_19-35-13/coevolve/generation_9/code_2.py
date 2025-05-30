import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    max_demand = demands.max()
    normalized_demands = demands / max_demand
    demand_matrix = torch.cat([normalized_demands.unsqueeze(0), normalized_demands.unsqueeze(1)], dim=0)
    demand_matrix[depot_index, depot_index] = 0

    # Initialize the heuristic matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Compute the heuristic values for each edge
    for i in range(n):
        for j in range(n):
            if i != j and demand_matrix[i][j] > 0:
                # Compute the heuristic value as a combination of the inverse distance and the inverse demand
                heuristics[i][j] = 1 / distance_matrix[i][j] + 1 / demand_matrix[i][j]

    return heuristics