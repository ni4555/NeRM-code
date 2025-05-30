import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate a base heuristic using the inverse of the demand (promising if low demand)
    base_heuristic = 1 / normalized_demands

    # Calculate the distance-based heuristic (undesirable if long distance)
    distance_heuristic = -distance_matrix

    # Combine the two heuristics using a simple linear combination
    # Note: The exact weights for the combination would need to be tuned
    combined_heuristic = base_heuristic + distance_heuristic

    # Clamp the values to ensure no negative values, which are undesirable
    combined_heuristic = torch.clamp(combined_heuristic, min=0)

    return combined_heuristic