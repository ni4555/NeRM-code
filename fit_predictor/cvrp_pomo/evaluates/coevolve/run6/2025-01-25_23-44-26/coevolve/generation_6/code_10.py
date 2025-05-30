import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Normalize customer demands by total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Inverse Distance Heuristic (IDH) - Promote closer customers
    idh_scores = 1.0 / distance_matrix

    # Demand penalty function - Penalize high-demand customers near capacity
    demand_penalty = demands / (demands.sum() + 1e-6)  # Avoid division by zero

    # Combine IDH and demand penalty scores
    combined_scores = idh_scores * normalized_demands * (1 - demand_penalty)

    # Ensure that the heuristics matrix has negative values for undesirable edges
    combined_scores[distance_matrix == 0] = 0  # Exclude depot itself
    combined_scores[combined_scores < 0] = 0
    combined_scores[combined_scores >= 0] -= combined_scores[combined_scores >= 0].min()  # Normalize to ensure positive values

    return combined_scores