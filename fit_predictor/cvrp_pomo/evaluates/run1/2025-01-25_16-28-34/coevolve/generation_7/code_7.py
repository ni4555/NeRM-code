import torch
import numpy as np
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the input tensors are on the same device and in the correct shape
    distance_matrix = distance_matrix.to(demands.device)
    demands = demands.to(demands.device)

    # Normalize the distance matrix by subtracting the minimum distance for each row (to account for symmetry)
    min_distance = distance_matrix.min(dim=1, keepdim=True)[0]
    distance_matrix -= min_distance

    # Normalize demands by dividing by the maximum demand (assuming that demand can be met by at least one vehicle)
    max_demand = demands.max()
    demands /= max_demand

    # Calculate a weight for each edge based on the inverse of the distance (to prefer shorter paths)
    # and the inverse of the demand (to prefer less demanding customers)
    edge_weights = 1 / (distance_matrix + 1e-10) + 1 / (demands + 1e-10)

    # Sum the weights for each customer, which represents the overall demand for each customer
    customer_weights = edge_weights.sum(dim=1)

    # Calculate a bonus for customers with a high demand-to-distance ratio
    # This encourages selecting edges that are close to high-demand customers
    demand_to_distance_ratio = demands / distance_matrix
    bonus = 1 / (demand_to_distance_ratio + 1e-10)

    # Combine the weights and the bonus to get the final heuristic values
    heuristic_values = customer_weights * bonus

    # Clip the values to ensure they are within a certain range to avoid overflow
    heuristic_values = torch.clamp(heuristic_values, min=-1e8, max=1e8)

    return heuristic_values

# Example usage
distance_matrix = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=torch.float32)
demands = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
heuristic_values = heuristics_v2(distance_matrix, demands)
print(heuristic_values)