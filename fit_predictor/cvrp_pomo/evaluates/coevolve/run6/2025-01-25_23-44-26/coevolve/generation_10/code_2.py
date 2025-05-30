import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand normalization
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the demands to promote assignments to customers with lower demand
    inverse_demands = 1 / (normalized_demands + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate the inverse distance heuristic (IDH)
    # For the purpose of this heuristic, we use the distance to the depot as the inverse distance
    # In a real scenario, this could be the inverse of the average distance to all other customers
    idh_values = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero

    # Combine IDH and demand-driven heuristics
    combined_heuristics = idh_values * inverse_demands

    # Define a penalty function for capacity constraints
    # Here we use a simple linear penalty proportional to the distance to the depot
    # In a real scenario, the penalty could be more sophisticated
    penalty = torch.clamp(distance_matrix, min=0)  # Ensure penalty is non-negative

    # Apply the penalty to the combined heuristics
    heuristics = combined_heuristics - penalty

    return heuristics