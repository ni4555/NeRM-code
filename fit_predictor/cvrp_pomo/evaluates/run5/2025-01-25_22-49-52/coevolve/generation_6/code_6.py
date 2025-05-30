import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the total vehicle capacity
    total_capacity = demands.sum()

    # Calculate the normalized demand for each customer
    normalized_demands = demands / total_capacity

    # Calculate the initial potential value for each edge
    potential_values = distance_matrix * normalized_demands

    # Apply node partitioning to the potential values
    # Here we use a simple approach where we sum the potential values of each edge
    # connected to a node and then normalize these sums to get a partitioning score
    node_partitioning = torch.sum(potential_values, dim=1) / torch.sum(potential_values, dim=0)

    # Apply demand relaxation by scaling the potential values by the demand
    demand_relaxed_potential_values = potential_values * demands

    # Apply path decomposition by considering only the edges with the highest potential values
    # We use a threshold to determine the highest potential values
    threshold = torch.max(demand_relaxed_potential_values)
    path_decomposed_potential_values = torch.where(demand_relaxed_potential_values >= threshold,
                                                 demand_relaxed_potential_values,
                                                 torch.zeros_like(demand_relaxed_potential_values))

    # Combine the effects of node partitioning, demand relaxation, and path decomposition
    combined_heuristic = node_partitioning * demand_relaxed_potential_values * path_decomposed_potential_values

    # Calculate the final heuristic values
    final_heuristic_values = combined_heuristic - threshold

    return final_heuristic_values