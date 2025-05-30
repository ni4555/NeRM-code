import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Normalize the demand vector
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Initialize the edge feasibility mask
    edge_feasibility_mask = torch.ones_like(distance_matrix, dtype=torch.float)

    # Calculate cumulative demand along the diagonal of the matrix
    cumulative_demand = torch.cumsum(normalized_demands, dim=0)

    # Adjust the edge feasibility mask based on cumulative demand
    edge_feasibility_mask = (cumulative_demand <= 1).float()

    # Calculate edge evaluation scores
    edge_scores = -distance_matrix  # Negative scores for distance

    # Incorporate demand and capacity constraints
    for i in range(edge_scores.shape[0]):
        for j in range(edge_scores.shape[1]):
            if i != j and edge_feasibility_mask[i, j] == 1:
                edge_scores[i, j] += normalized_demands[i] + normalized_demands[j]

    return edge_scores