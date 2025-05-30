import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the sum of all demands
    total_demand = demands.sum()

    # Normalize the demands to the total vehicle capacity
    normalized_demands = demands / total_demand

    # Calculate the difference between 1 and the normalized demand for each node
    demand_difference = 1 - normalized_demands

    # Create a mask for edges that are within a certain threshold of the nearest demand
    # This is a simple heuristic to consider edges that have demand difference close to 1
    threshold = 0.2
    demand_diff_mask = demand_difference.abs() < threshold

    # Calculate the distance to the nearest demand for each node
    # This is a simple heuristic that considers the node closer to demand as more promising
    distance_to_nearest_demand = torch.min(distance_matrix, dim=1, keepdim=True)[0]

    # Normalize the distances by the maximum distance in the matrix
    normalized_distances = distance_to_nearest_demand / torch.max(distance_to_nearest_demand)

    # Combine the heuristics using the demand difference and normalized distances
    heuristics_matrix = (demand_diff_mask * demand_difference) + (1 - demand_diff_mask) * normalized_distances

    return heuristics_matrix