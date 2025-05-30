import random
import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Step 1: Use demand relaxation to adjust the heuristics
    # We calculate the potential load for each edge based on demand
    load_potential = demands.unsqueeze(1) + demands.unsqueeze(0)
    # Normalize by total vehicle capacity
    load_potential /= torch.sum(demands)
    # Adjust heuristics for demand relaxation
    heuristics_matrix += load_potential

    # Step 2: Use dynamic window approach to adjust for dynamic changes
    # Here, we simulate the dynamic window by considering current demand
    # Assuming that the distance_matrix is updated dynamically and we have the current demand
    # For simplicity, we will not actually change the distance_matrix, but simulate the effect
    # with a dynamic factor

    # Define a dynamic factor that changes with time (for simplicity, we will use a constant factor)
    dynamic_factor = 0.5
    # Adjust heuristics for dynamic window
    heuristics_matrix *= dynamic_factor

    # Step 3: Apply node partitioning for path decomposition
    # We will create a partitioning matrix based on a threshold to group nodes
    # For simplicity, we use a fixed threshold
    threshold = 0.3  # This threshold can be dynamically adjusted
    # Create a partitioning matrix based on load potential
    partitioning_matrix = (load_potential > threshold).float()

    # Adjust heuristics for partitioning
    # Nodes in the same partition have a lower cost
    for i in range(n):
        for j in range(n):
            if partitioning_matrix[i, j] == 1:
                heuristics_matrix[i, j] -= 1  # Lower the cost for intra-partition edges

    # Step 4: Apply multi-objective evolutionary algorithm (MEEA) principles
    # Since this is a heuristic, we simulate MEEA principles by adjusting the heuristics
    # based on some random selection that mimics genetic diversity
    # For simplicity, we use a random factor
    random_factor = torch.rand_like(distance_matrix) * 0.1
    heuristics_matrix -= random_factor  # Negative values for undesirable edges

    return heuristics_matrix