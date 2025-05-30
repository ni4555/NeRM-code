import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the cost of each edge (negative cost for undesirable edges)
    # The cost is a function of the distance and the difference in normalized demand
    cost_matrix = distance_matrix * (1 - torch.abs(normalized_demands.unsqueeze(1) - normalized_demands.unsqueeze(0)))

    # Use a temperature-based approach to encourage exploration of promising edges
    # This can be adjusted by a temperature parameter
    temperature = 0.5
    cost_matrix = torch.exp(-cost_matrix / temperature)

    # Normalize the cost matrix so that each edge has a probability of being included
    cost_matrix = cost_matrix / cost_matrix.sum(dim=1, keepdim=True)

    # Sample the edges based on their probabilities to get the heuristic matrix
    heuristic_matrix = torch.multinomial(cost_matrix, num_samples=1, replacement=True)

    return heuristic_matrix