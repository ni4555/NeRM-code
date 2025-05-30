import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_capacity = demands.sum()
    demand_normalized = demands / total_capacity

    # Greedy assignment based on normalized demand
    demand_sorted_indices = torch.argsort(demand_normalized)[::-1]
    greedy_assignment = torch.zeros(n, dtype=torch.int64)
    load = torch.zeros(n, dtype=torch.float64)

    for i in range(n):
        # Find the best unassigned customer
        best_customer = torch.argmax((demand_sorted_indices != i) & (load < 1))
        greedy_assignment[i] = demand_sorted_indices[best_customer]
        load[best_customer] += 1

    # Calculate heuristics based on distance and demand
    heuristics = torch.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(n):
            if i != j and demand_normalized[i] > 0:
                heuristics[i, j] = -distance_matrix[i, j]

    # Local search to refine the initial solutions
    for _ in range(100):  # Number of iterations for local search
        improved = False
        for i in range(n):
            for j in range(n):
                if i != j and demand_normalized[i] > 0 and heuristics[i, j] > 0:
                    # Swap customers i and j
                    temp_demand = demands[i]
                    demands[i] = demands[j]
                    demands[j] = temp_demand

                    # Recalculate demand normalization
                    load = torch.zeros(n, dtype=torch.float64)
                    for k in range(n):
                        best_customer = torch.argmax((demand_sorted_indices != k) & (load < 1))
                        greedy_assignment[k] = demand_sorted_indices[best_customer]
                        load[demand_sorted_indices[best_customer]] += 1

                    # Calculate new heuristics
                    new_heuristics = torch.zeros_like(distance_matrix)
                    for k in range(n):
                        for l in range(n):
                            if k != l and demand_normalized[k] > 0:
                                new_heuristics[k, l] = -distance_matrix[k, l]

                    # Restore original demand if improvement is not found
                    if new_heuristics[i, j] < heuristics[i, j]:
                        heuristics = new_heuristics
                        improved = True
                    else:
                        temp_demand = demands[i]
                        demands[i] = demands[j]
                        demands[j] = temp_demand

        if not improved:
            break

    return heuristics