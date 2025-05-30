import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Demand normalization
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Evaluate each edge
    for i in range(distance_matrix.size(0)):
        for j in range(distance_matrix.size(1)):
            if i != j:
                # Calculate the edge evaluation based on distance and normalized demand
                edge_evaluation = distance_matrix[i][j] * normalized_demands[i]
                # Assign the evaluation to the corresponding edge in the heuristic matrix
                heuristic_matrix[i][j] = edge_evaluation

    return heuristic_matrix