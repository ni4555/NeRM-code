import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the inverse distance heuristics
    # Since we're using the inverse distance, we should have a positive heuristic for the shorter distances
    # We'll subtract the demand-sensitive term to penalize overloading
    idh = 1 / (distance_matrix ** 2)

    # Calculate the demand-sensitive penalty mechanism
    # The idea is to penalize routes that come close to the vehicle's capacity
    # Here we use a simple approach where we penalize by the ratio of (vehicle capacity - current demand) to vehicle capacity
    # This ensures that as a vehicle approaches its capacity, the heuristic value decreases
    demand_penalty = 1 - normalized_demands

    # Combine the heuristics
    # Note: We add the demand penalty to the IDH because we want to encourage routes that have more space in the vehicle
    combined_heuristics = idh + demand_penalty

    return combined_heuristics

# Example usage:
# Create a sample distance matrix and demands
distance_matrix = torch.tensor([[0, 2, 6, 8], [2, 0, 1, 5], [6, 1, 0, 3], [8, 5, 3, 0]], dtype=torch.float32)
demands = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

# Get the heuristics matrix
heuristic_matrix = heuristics_v2(distance_matrix, demands)

print(heuristic_matrix)