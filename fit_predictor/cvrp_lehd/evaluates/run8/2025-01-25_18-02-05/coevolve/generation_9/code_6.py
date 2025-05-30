import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate a baseline score for all edges that is a combination of distance and demand
    # A negative coefficient is used to give higher priority to closer edges with lower demand
    # These coefficients might need to be tuned based on the specifics of the problem
    alpha = -0.5  # Coefficient for distance
    beta = 0.5     # Coefficient for demand (normalized by total capacity)
    baseline_score = alpha * distance_matrix + beta * normalized_demands

    # Introduce negative scores for the depot-to-depot edge and edges that exceed vehicle capacity
    negative_edges = distance_matrix == 0
    capacity_exceeding_edges = (distance_matrix != 0) * (demands > 1)  # Assuming demands are normalized and should not exceed 1
    baseline_score[negative_edges] = -float('inf')  # Make depot-to-depot edge extremely undesirable
    baseline_score[capacity_exceeding_edges] = -float('inf')  # Make capacity-exceeding edges extremely undesirable

    return baseline_score