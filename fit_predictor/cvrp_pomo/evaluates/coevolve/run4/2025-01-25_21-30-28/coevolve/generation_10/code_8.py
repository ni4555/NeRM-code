import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are on the same device and of the same dtype
    distance_matrix = distance_matrix.to(demands.device).to(demands.dtype)
    demands = demands.to(distance_matrix.device).to(distance_matrix.dtype)

    # Calculate the inverse of the demands to represent the urgency of service
    inverse_demands = 1 / demands

    # Apply the Normalization heuristic
    # This heuristic assigns a value based on the distance to the nearest node
    # and the urgency of the demand.
    min_distances = torch.min(distance_matrix, dim=1, keepdim=True)[0]
    normalization_heuristic = (inverse_demands * min_distances).unsqueeze(1)

    # Apply the Inverse Distance heuristic
    # This heuristic assigns a value inversely proportional to the distance.
    inverse_distance_heuristic = (inverse_demands * (distance_matrix ** 2)).unsqueeze(1)

    # Combine the two heuristics using a weighted sum (weights can be adjusted)
    # For simplicity, we'll use equal weights, but these can be tuned.
    combined_heuristic = (normalization_heuristic + inverse_distance_heuristic) / 2

    # Subtract the combined heuristic from the demands to get negative values for undesirable edges
    # and positive values for promising ones.
    heuristics = demands - combined_heuristic

    return heuristics