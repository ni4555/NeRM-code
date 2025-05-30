import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Problem-specific Local Search
    # Calculate a simple heuristic based on the distance to the depot and the customer demand
    for i in range(1, n):  # Skip the depot node
        # Promote edges based on lower distance and demand
        heuristic_matrix[i, 0] = -distance_matrix[i, 0] - demands[i]

    # Adaptive PSO Population Management
    # For simplicity, we will use a basic PSO-inspired heuristic that promotes edges closer to the depot
    # and with lower demand, which could be considered as a "fitness" for PSO.
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # Promote edges closer to the depot and with lower demand
                heuristic_matrix[i, j] = -distance_matrix[i, j] - demands[j]

    # Dynamic Tabu Search with Adaptive Cost Function
    # For simplicity, we will use a tabu list that bans the last used edges
    # This is a placeholder for a more complex tabu search mechanism
    tabu_list = set()
    for i in range(1, n):
        for j in range(1, n):
            if i != j and (i, j) not in tabu_list:
                # Increase the heuristic value for edges not in the tabu list
                heuristic_matrix[i, j] += 1

    return heuristic_matrix