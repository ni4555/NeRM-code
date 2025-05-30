import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are on the same device and tensor type
    distance_matrix = distance_matrix.to(demands.device).to(demands.dtype)
    demands = demands.to(distance_matrix.device).to(distance_matrix.dtype)

    # Get the number of customers (excluding the depot)
    num_customers = distance_matrix.shape[0] - 1

    # Compute the total vehicle capacity
    total_capacity = demands.sum()

    # Normalize customer demands to the total vehicle capacity
    normalized_demands = demands / total_capacity

    # Compute the heuristic values for each edge
    # The heuristic function used here is a simple one: the negative of the distance
    # multiplied by the normalized demand of the destination node.
    # This is a placeholder for the actual heuristic which may be more complex.
    heuristics = -distance_matrix * normalized_demands

    # Apply a demand normalization technique to ensure equitable solution evaluation
    # Here we use the sum of the normalized demands to scale the heuristic values
    normalized_sum = heuristics.sum(dim=1, keepdim=True)
    heuristics /= normalized_sum

    # Apply a constraint-aware allocation strategy to optimize capacity utilization
    # and prevent overloading. This step could involve more complex logic and is
    # simplified here for demonstration purposes.
    # We ensure that the sum of demands in any vehicle does not exceed the capacity
    # by clamping the sum to the total capacity.
    vehicle_demands = heuristics.sum(dim=1, keepdim=True)
    vehicle_demands = torch.clamp(vehicle_demands, min=0, max=total_capacity)

    # Adjust the heuristic values based on the vehicle demands
    # Here we use a simple penalty mechanism: the more the demand of a vehicle,
    # the higher the penalty for its edges.
    penalty = (vehicle_demands / total_capacity) * 1000  # arbitrary penalty factor
    heuristics -= penalty

    return heuristics