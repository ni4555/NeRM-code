import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity

    # Greedy assignment based on normalized demand
    demand_normalized = normalized_demands / normalized_demands.sum()
    demand_normalized = demand_normalized.view(1, -1)
    initial_assignment = torch.argmax(distance_matrix * demand_normalized, dim=1)

    # Calculate initial heuristics based on distances and demands
    heuristics = distance_matrix[torch.arange(n), initial_assignment]

    # Local search to refine the solution
    for _ in range(10):  # Number of iterations for local search
        for i in range(1, n):  # Skip the depot node
            for j in range(1, n):  # Skip the depot node
                if i != j:
                    # Swap customers between routes
                    new_assignment = initial_assignment.clone()
                    new_assignment[i] = initial_assignment[j]
                    new_assignment[j] = initial_assignment[i]

                    # Calculate the cost of the new assignment
                    new_cost = distance_matrix[torch.arange(n), new_assignment].sum()

                    # If the new cost is better, update the assignment
                    if new_cost < heuristics.sum():
                        initial_assignment = new_assignment
                        heuristics = distance_matrix[torch.arange(n), initial_assignment]

    # Normalize heuristics to have a range between 0 and 1
    heuristics = (heuristics - heuristics.min()) / (heuristics.max() - heuristics.min())

    return heuristics