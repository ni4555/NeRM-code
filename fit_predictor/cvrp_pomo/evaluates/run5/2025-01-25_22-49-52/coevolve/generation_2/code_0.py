import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Constants
    depot_index = 0
    n = distance_matrix.size(0)
    total_capacity = demands.sum()

    # Node partitioning
    node_partitioning = torch.zeros_like(demands)
    node_partitioning[1:] = (demands[1:] / demands[1:].sum()).unsqueeze(0)
    partition_indices = torch.argmax(node_partitioning, dim=0)

    # Dynamic window technique
    dynamic_window = torch.arange(n)
    dynamic_window[partition_indices] += 1  # Adjust indices based on partitioning

    # Demand relaxation
    relaxed_demands = (demands - 1) / demands

    # Multi-objective evolutionary algorithm heuristic (simplified version)
    # Calculate the "promise" score based on distance and demand relaxation
    distance_heuristic = -distance_matrix
    demand_heuristic = relaxed_demands * 0.5  # Assuming a simple weighting for demonstration
    multi_objective_score = distance_heuristic + demand_heuristic

    # Adjusting scores with dynamic window
    adjusted_scores = multi_objective_score[dynamic_window]

    # Path decomposition - ignore depot edges for now
    adjusted_scores[torch.arange(n) == depot_index] = -float('inf')

    # Return the heuristic scores
    return adjusted_scores