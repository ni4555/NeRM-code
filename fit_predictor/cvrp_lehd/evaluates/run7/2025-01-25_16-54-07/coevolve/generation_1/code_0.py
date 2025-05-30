import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum demand per distance for each edge
    max_demand_per_distance = torch.max(demands[1:], dim=0)[0]  # Exclude the depot node
    max_demand_per_distance[distance_matrix == 0] = float('-inf')  # No distance for the depot node

    # Calculate the normalized demand per distance
    normalized_demand_per_distance = max_demand_per_distance / (distance_matrix + 1e-6)  # Add a small value to avoid division by zero

    # Calculate the heuristic values based on normalized demand and distance
    heuristics_values = -normalized_demand_per_distance

    return heuristics_values