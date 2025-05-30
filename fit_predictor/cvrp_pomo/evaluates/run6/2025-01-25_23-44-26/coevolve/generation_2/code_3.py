import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize demands to be within [0, 1]
    normalized_demands = (demands / total_capacity).to(torch.float32)

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix, dtype=torch.float32)

    # Calculate the heuristic values for each edge
    # This is a simple example using the inverse of the demand ratio as the heuristic
    # A more complex heuristic can be designed here
    heuristic_matrix = 1 / (normalized_demands.unsqueeze(1) + normalized_demands.unsqueeze(0))

    # Discourage high distance edges by subtracting them from the heuristic
    # This encourages the algorithm to avoid long routes
    heuristic_matrix -= distance_matrix

    # Set the diagonal to -infinity to avoid self-loops
    torch.fill_diagonal_(heuristic_matrix, float('-inf'))

    return heuristic_matrix