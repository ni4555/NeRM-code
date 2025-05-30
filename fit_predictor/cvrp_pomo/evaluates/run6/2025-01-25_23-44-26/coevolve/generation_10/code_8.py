import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demands by the total vehicle capacity (assuming this is a single vehicle scenario)
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse of the demands to create a demand-driven heuristic
    inverse_demands = 1 / (normalized_demands + 1e-6)  # Adding a small constant to avoid division by zero

    # Calculate the inverse distance heuristic (IDH) using the distance matrix
    # Here, we assume that the smaller the distance, the higher the heuristic value
    idh_values = 1 / (distance_matrix + 1e-6)  # Adding a small constant to avoid division by zero

    # Combine the inverse demands and IDH into a single heuristic matrix
    # The weight for inverse demands can be adjusted based on the problem characteristics
    weight_inverse_demands = 0.5
    weight_idh = 0.5
    combined_heuristics = weight_inverse_demands * inverse_demands + weight_idh * idh_values

    # Apply a penalty for edges that are close to exceeding the vehicle's capacity
    # This can be adjusted based on the specific capacity constraint requirements
    penalty_threshold = 0.95  # Assuming the vehicle capacity is 100% and we penalize when reaching 95%
    capacity_penalty = (1 - normalized_demands) * (penalty_threshold - normalized_demands)
    combined_heuristics = combined_heuristics - capacity_penalty

    return combined_heuristics