import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Greedy assignment based on normalized demand
    demand_sorted_indices = torch.argsort(normalized_demands)
    demand_sorted_distances = torch.gather(distance_matrix, 1, demand_sorted_indices.unsqueeze(0))

    # Initialize the heuristic matrix with negative infinity
    heuristic_matrix = torch.full((n, n), float('-inf'))

    # Set the cost for each customer to its nearest neighbor in the sorted list
    for i in range(n):
        if i != 0:  # Skip the depot
            nearest_neighbor = torch.argmin(demand_sorted_distances[i])
            heuristic_matrix[i, nearest_neighbor] = distance_matrix[i, nearest_neighbor]

    # Set the cost from the last customer back to the depot
    heuristic_matrix[torch.arange(1, n), 0] = distance_matrix[torch.arange(1, n), 0]

    # Local search to refine initial solutions
    for _ in range(n):  # Number of iterations can be tuned
        improved = False
        for i in range(n):
            for j in range(n):
                if i != j and j != 0 and (i < n - 1 or i == 0):  # Skip the depot
                    cost_without_i = heuristic_matrix[i, 0] + heuristic_matrix[j, 0]
                    cost_with_i_j = heuristic_matrix[i, j] + heuristic_matrix[j, 0]
                    if cost_with_i_j < cost_without_i:
                        heuristic_matrix[i, j] = cost_with_i_j
                        improved = True
        if not improved:
            break

    return heuristic_matrix