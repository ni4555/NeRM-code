import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix by the maximum distance to ensure that all values are in a manageable range
    max_distance = torch.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance

    # Calculate the cumulative demand along each row (from the depot to other nodes)
    cumulative_demand = torch.cumsum(demands, dim=0)

    # Calculate the cumulative distance from the depot to each node
    cumulative_distance = torch.cumsum(normalized_distance_matrix, dim=1)

    # Calculate the combined heuristics value for each edge
    # The heuristic is a weighted sum of the inverse distance and the ratio of the demand to the cumulative demand
    # The weights are chosen to prioritize both distance minimization and demand relaxation
    alpha = 0.5  # Weight for distance
    beta = 0.5   # Weight for demand relaxation
    combined_heuristics = alpha * (1 / normalized_distance_matrix) + beta * (demands / cumulative_demand)

    # Apply a threshold to the combined heuristics to ensure that only promising edges are included
    threshold = 0.2  # Threshold for heuristics value
    heuristics = torch.where(combined_heuristics > threshold, combined_heuristics, torch.zeros_like(combined_heuristics))

    return heuristics