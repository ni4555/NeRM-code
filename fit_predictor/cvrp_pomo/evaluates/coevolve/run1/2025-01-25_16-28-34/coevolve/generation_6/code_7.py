import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    demands = demands.to(distance_matrix.dtype)
    distance_matrix = distance_matrix.to(distance_matrix.dtype)

    # Problem-specific Local Search
    # Calculate the initial heuristic values based on the sum of distances to the depot
    initial_heuristic = -torch.sum(distance_matrix, dim=1)

    # Adaptive PSO Population Management
    # For simplicity, we'll use a simple heuristic based on the PSO principle
    # by considering the inverse of the distance to the depot as a proxy for PSO's fitness
    # This is a simplified representation and does not reflect a true PSO algorithm
    pso_heuristic = 1.0 / (distance_matrix[depot_index] + 1e-6)

    # Dynamic Tabu Search with Adaptive Cost Function
    # Calculate a heuristic based on the total distance to the depot
    tabu_heuristic = -torch.sum(distance_matrix, dim=1)

    # Combine heuristics with weights to represent their contribution
    # Note: Weights should be tuned for the specific problem and are chosen arbitrarily here
    combined_heuristic = 0.5 * (initial_heuristic + pso_heuristic + tabu_heuristic)

    return combined_heuristic