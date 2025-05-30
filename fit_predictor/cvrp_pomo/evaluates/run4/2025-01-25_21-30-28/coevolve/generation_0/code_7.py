import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by the maximum demand to ensure the relative scale
    max_demand = torch.max(demands)
    normalized_demands = demands / max_demand

    # Calculate the negative of the demand, which is a heuristic for undesirable edges
    negative_demand = -normalized_demands

    # Calculate a simple heuristic based on the inverse of the demand, promoting edges with lower demand
    inverse_demand = 1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero

    # Calculate a simple heuristic based on the distance, which can be used to penalize longer distances
    distance_penalty = distance_matrix / (torch.sum(distance_matrix, dim=1) + 1e-6)  # Avoid division by zero

    # Combine heuristics to create the final heuristic matrix
    combined_heuristics = negative_demand + inverse_demand + distance_penalty

    return combined_heuristics