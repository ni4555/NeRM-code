import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are both of the same size
    assert distance_matrix.shape == demands.shape, "Distance matrix and demands must have the same shape."

    # Initialize the potential values matrix with zeros
    potential_values = torch.zeros_like(distance_matrix)

    # Get the total vehicle capacity (assuming it's a scalar for simplicity)
    total_capacity = demands.sum()

    # Calculate the maximum load that can be transported on each edge
    max_load_on_edge = demands

    # Calculate the potential value for each edge
    # For this heuristic, we assume that the potential value is the maximum load that can be transported on the edge
    potential_values = max_load_on_edge

    # To make the heuristic more promising, we can subtract the distance (or some other factor) from the potential value
    # This step is optional and can be adjusted depending on the specifics of the problem
    potential_values -= distance_matrix

    # Normalize the potential values to ensure they are within the desired range
    # In this example, we'll ensure that all negative values are zero and all positive values are scaled to be positive
    potential_values = torch.clamp(potential_values, min=0)

    return potential_values