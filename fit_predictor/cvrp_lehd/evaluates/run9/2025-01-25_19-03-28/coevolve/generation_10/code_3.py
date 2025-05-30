import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize demands to fit within the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Initialize a tensor to store the heuristic values
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the heuristic values for each edge
    # 1. The heuristic is a combination of demand-based and distance-based factors
    # Demand-based heuristic: higher demand implies a more promising edge
    demand_heuristic = 1 - normalized_demands

    # Distance-based heuristic: shorter distances are more promising
    distance_heuristic = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero

    # Combine the demand and distance heuristics
    heuristic_matrix = demand_heuristic * distance_heuristic

    return heuristic_matrix