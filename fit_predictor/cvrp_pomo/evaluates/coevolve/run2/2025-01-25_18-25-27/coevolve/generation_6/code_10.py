import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check that the demands tensor is not empty
    if demands.numel() == 0:
        return torch.zeros_like(distance_matrix)

    # Normalize the demands to the range of [0, 1]
    max_demand = torch.max(demands)
    demands_normalized = demands / max_demand

    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Fill the heuristics matrix based on the normalized demands
    # We use a simple heuristic where edges are marked as promising if their demands do not exceed the capacity
    # This is a placeholder heuristic; the actual heuristic should be adapted to the problem's specifics
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            # The depot is node 0 and its demand is 0, we skip it
            if i == 0 or j == 0:
                continue

            # Negative value for undesirable edges (demand sum exceeds capacity)
            heuristics_matrix[i, j] = -demands_normalized[i] - demands_normalized[j]

            # We can add a threshold to make the values positive if they're promising
            # For instance, if demands_normalized[i] + demands_normalized[j] < 1:
            #     heuristics_matrix[i, j] = 1.0 - demands_normalized[i] - demands_normalized[j]
            # else:
            #     heuristics_matrix[i, j] = -1.0 * (demands_normalized[i] + demands_normalized[j])

    return heuristics_matrix