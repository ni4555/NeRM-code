import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the negative weighted distance matrix
    # We use negative values to indicate shorter distances as more promising
    negative_distance_matrix = -distance_matrix

    # Integrate demand-based prioritization
    # We add a term that prioritizes edges with lower demand
    demand_weighted_matrix = negative_distance_matrix + normalized_demands.unsqueeze(1) * demands.unsqueeze(0)

    # Integrate proximity-based route planning
    # We add a term that prioritizes edges closer to the depot
    depot_index = 0
    distance_to_depot = torch.zeros_like(distance_matrix)
    distance_to_depot.fill_(float('inf'))
    distance_to_depot[depot_index] = 0
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            distance_to_depot[j] = torch.min(distance_to_depot[j], distance_to_depot[i] + distance_matrix[i][j])
    distance_to_depot = -distance_to_depot
    proximity_weighted_matrix = demand_weighted_matrix + distance_to_depot.unsqueeze(1) * distance_to_depot.unsqueeze(0)

    # Dynamic load balancing
    # We add a term that balances the load across the edges
    load_balance_matrix = proximity_weighted_matrix.clone()
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            load_balance_matrix[i][j] += (demands[i] - demands[j]) * (demands[i] - demands[j])

    return load_balance_matrix