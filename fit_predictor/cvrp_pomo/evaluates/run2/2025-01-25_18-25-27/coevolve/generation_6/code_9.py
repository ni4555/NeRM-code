import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the normalized demands
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Compute the heuristic values
    # The heuristic function could be based on the ratio of customer demand to distance
    # Here we use a simple heuristic where we multiply the normalized demand by the distance
    # This heuristic is a simple example and may need to be adjusted for a more complex scenario
    heuristic_matrix = normalized_demands.unsqueeze(1) * distance_matrix.unsqueeze(0)

    # Negative values represent undesirable edges, positive values represent promising ones
    # In this example, we consider higher demands and shorter distances as more promising
    # You can adjust the signs or the heuristic function to suit your needs

    return heuristic_matrix