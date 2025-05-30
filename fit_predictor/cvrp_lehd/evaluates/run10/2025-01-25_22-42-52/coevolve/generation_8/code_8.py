import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands to fit the total vehicle capacity (which we'll assume to be 1 in this case)
    normalized_demands = demands / demands.sum()

    # Initialize the heuristic matrix with a high value (for undesirable edges)
    heuristic_matrix = torch.ones_like(distance_matrix) * float('inf')

    # Calculate the heuristic for each edge
    # The heuristic is a combination of distance and normalized demand
    # A higher demand and/or shorter distance increases the heuristic value
    heuristic_matrix = (1 - normalized_demands.unsqueeze(0) * normalized_demands.unsqueeze(1)) * distance_matrix

    # Set diagonal values to a very low value to avoid including the depot in the solution
    torch.fill_diagonal_(heuristic_matrix, -float('inf'))

    # Apply some logic to convert the matrix to a desired scale (-infinity for undesirable and +infinity for promising)
    # We will make the diagonal undesirable by setting negative infinity to the diagonal values
    # For all other edges, we want the negative values for undesirable edges
    heuristic_matrix[heuristic_matrix < 0] = -float('inf')

    return heuristic_matrix