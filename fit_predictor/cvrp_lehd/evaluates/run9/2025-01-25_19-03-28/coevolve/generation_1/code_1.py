import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix to have a scale that is easier to work with
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the sum of demands
    total_demand = torch.sum(demands)

    # Calculate the potential value for each edge
    potential_value = normalized_distance_matrix * demands

    # Normalize the potential value to have a scale that is easier to work with
    max_potential_value = torch.max(potential_value)
    normalized_potential_value = potential_value / max_potential_value

    # Calculate the heuristic value for each edge
    # Here we use a simple heuristic that is a combination of normalized distance and potential value
    # We subtract a small constant to make all values negative for undesirable edges
    heuristic_value = normalized_distance_matrix - normalized_potential_value

    # Ensure that all values are within the desired range (-1, 1)
    # We add 1 to shift the range from (-1, 1) to (0, 2)
    # Then we divide by 2 to scale it back to (-1, 1)
    heuristic_value = (heuristic_value + 1) / 2

    return heuristic_value