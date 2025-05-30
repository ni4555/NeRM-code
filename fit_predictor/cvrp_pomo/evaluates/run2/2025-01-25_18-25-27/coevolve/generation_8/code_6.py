import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    vehicle_capacity = demands.sum()
    normalized_demands = demands / vehicle_capacity

    # Step 1: Calculate initial heuristic based on normalized demands
    demand_heuristic = -normalized_demands[None, :] @ distance_matrix.T

    # Step 2: Apply 2-opt heuristic for capacity-aware route optimization
    # Note: This is a simplified version of the 2-opt heuristic that doesn't change the distance matrix
    # but rather adjusts the heuristic based on potential savings from improving the route
    for i in range(1, n):
        for j in range(i + 2, n):
            if i + 1 < j:  # Ensure the subtour is not trivial
                # Calculate the savings if the subtour from i, j-1, ..., i+1, j is optimized
                savings = -torch.abs((distance_matrix[i, j] - distance_matrix[i, j - 1] -
                                      distance_matrix[i + 1, j] + distance_matrix[i + 1, j - 1]))
                demand_heuristic[i, j] += savings

    # Step 3: Apply swap-insertion heuristic
    for i in range(1, n):
        for j in range(i + 2, n):
            # Consider swapping customer i with customer j
            # Here, we simulate this by considering a change in heuristic for swap
            swap_savings = -torch.abs((distance_matrix[i, j] - distance_matrix[i, j - 1] -
                                      distance_matrix[i + 1, j] + distance_matrix[i + 1, j - 1]))
            demand_heuristic[i, j] += swap_savings

    # Step 4: Incorporate real-time penalties to prevent overloading
    # This step requires information about the current route load, which is not provided here.
    # Assuming that the distance_matrix contains both distances and the load at each node:
    # distance_matrix = torch.cat([distance_matrix, load_matrix], dim=1)
    # for i in range(1, n):
    #     for j in range(i + 2, n):
    #         # Assume the last column of distance_matrix contains load information
    #         load_penalty = max(0, distance_matrix[i, j].item() - vehicle_capacity)
    #         demand_heuristic[i, j] -= load_penalty

    return demand_heuristic