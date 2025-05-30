import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse distance heuristic: edges with smaller distances are more promising
    inverse_distance = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Demand normalization heuristic: edges with higher normalized demand are more promising
    demand_heuristic = normalized_demands.unsqueeze(1) * normalized_demands.unsqueeze(0)

    # Combine both heuristics by taking the minimum (which will be more negative for less promising edges)
    combined_heuristic = torch.min(inverse_distance, demand_heuristic)

    return combined_heuristic