import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Calculate the cumulative sum of demands to create a demand-based heuristic
    cumulative_demand = torch.cumsum(demands, dim=0)
    # Normalize the cumulative demand to create a demand-based heuristic
    demand_heuristic = cumulative_demand / cumulative_demand[-1]
    # Calculate the distance-based heuristic
    distance_heuristic = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    # Combine the two heuristics, giving more weight to demand-based heuristic
    combined_heuristic = demand_heuristic * 0.5 + distance_heuristic * 0.5
    return combined_heuristic