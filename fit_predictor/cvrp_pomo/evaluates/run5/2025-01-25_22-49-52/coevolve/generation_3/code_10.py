import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Check if the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    if not torch.isclose(total_capacity, 1.0):
        raise ValueError("Demands must be normalized by the total vehicle capacity.")

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Node partitioning: Create a partition of nodes based on their demands
    sorted_indices = torch.argsort(demands)
    threshold = demands[sorted_indices[int(0.5 * len(demands))]]
    partition = torch.where(demands > threshold)[0]

    # Demand relaxation: Relax the demands slightly to allow for more flexibility
    relaxed_demands = demands * 0.95

    # Path decomposition: Start from the depot and recursively explore paths
    def explore_paths(current_node, path, visited):
        visited[current_node] = True
        path.append(current_node)
        if len(path) == 1:
            return distance_matrix[0, current_node]
        min_cost = float('inf')
        for next_node in range(len(demands)):
            if not visited[next_node] and relaxed_demands[next_node] <= vehicle_capacity:
                cost = distance_matrix[current_node, next_node] + explore_paths(next_node, path.copy(), visited)
                min_cost = min(min_cost, cost)
        path.pop()
        visited[current_node] = False
        return min_cost

    # Dynamic window approach: Use a sliding window to update the heuristic matrix
    vehicle_capacity = 1.0  # Assuming the vehicle capacity is 1 for normalization
    for _ in range(5):  # Number of iterations can be tuned
        visited = torch.zeros_like(demands, dtype=torch.bool)
        path = []
        total_cost = explore_paths(0, path, visited)
        for node in path:
            for next_node in range(len(demands)):
                if not visited[next_node] and relaxed_demands[next_node] <= vehicle_capacity:
                    heuristic_matrix[node, next_node] = -total_cost

    return heuristic_matrix