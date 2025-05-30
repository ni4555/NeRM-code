import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative of the distance matrix as a heuristic for undesirable edges
    negative_distance_matrix = -distance_matrix

    # Calculate the demand-based heuristic for desirable edges
    demand_based_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Combine the two heuristics by element-wise addition
    combined_heuristic = negative_distance_matrix + demand_based_heuristic

    # Clip the values to ensure they are within the desired range (e.g., -max_demand to max_demand)
    max_demand = demands.max()
    combined_heuristic = torch.clamp(combined_heuristic, min=-max_demand, max=max_demand)

    return combined_heuristic