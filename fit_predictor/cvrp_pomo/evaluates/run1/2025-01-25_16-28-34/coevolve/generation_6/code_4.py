import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Problem-specific Local Search: Initialize a promisingness matrix with negative values
    promisingness_matrix = -torch.ones_like(distance_matrix)

    # Calculate initial load for each customer
    load = demands.clone()

    # Define a threshold for load imbalance
    load_threshold = 0.1

    # Initialize a variable to keep track of the maximum load
    max_load = demands[0]

    # Iterate over all edges to assign them based on load and distance
    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if adding this customer would cause a significant load imbalance
                new_load = load[i] + demands[j]
                load_diff = abs(new_load - load[i])

                # Calculate the distance-based score
                distance_score = -distance_matrix[i, j]

                # Adjust score based on load
                if load_diff <= load_threshold * max_load:
                    load_score = -0.5 * load_diff
                else:
                    load_score = -load_diff

                # Update the promisingness matrix
                promisingness_matrix[i, j] = distance_score + load_score

                # Update the maximum load
                max_load = max(max_load, new_load)

    return promisingness_matrix