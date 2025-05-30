import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized
    demands = demands / demands.sum()

    # Initialize a tensor of the same shape as distance_matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the maximum possible demand that a vehicle can carry without exceeding its capacity
    max_demand = demands.sum()

    # Iterate over each node pair
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Exclude the depot from itself
                # If the distance is finite (not a placeholder value)
                if torch.isfinite(distance_matrix[i, j]):
                    # Calculate the potential benefit of traveling from node i to node j
                    # This is the demand of node j minus the fraction of the total demand
                    # already carried by the vehicle on the way from the depot to node i
                    benefit = demands[j] - demands[:i+1].sum()
                    # If the benefit is positive, this edge is promising
                    heuristics[i, j] = benefit

    return heuristics