import torch
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n_nodes = distance_matrix.shape[0]
    heuristics = -torch.ones_like(distance_matrix)
    demand_cumsum = torch.cumsum(demands, dim=0)
    remaining_capacity = demands[1:]  # Exclude the depot node from capacity calculation
    
    # Normalize distance matrix
    normalized_distance_matrix = distance_matrix / torch.sum(distance_matrix[:, 0])
    
    for start_node in range(1, n_nodes):
        for destination_node in range(start_node + 1, n_nodes):
            total_demand = demand_cumsum[destination_node] - demand_cumsum[start_node]
            if total_demand <= remaining_capacity[start_node]:
                distance = normalized_distance_matrix[start_node, destination_node]
                heuristics[start_node, destination_node] = distance
                remaining_capacity[start_node] -= total_demand
                # Update the heuristic for the return to the depot
                heuristics[start_node, 0] = normalized_distance_matrix[start_node, 0]
                # Break inner loop if no capacity left for further nodes
                if remaining_capacity[start_node] <= 0:
                    break
    
    # Exploit symmetry by adding the transposed matrix
    heuristics = heuristics + heuristics.t()
    
    # Balance criteria: Normalize by total distance to depot
    total_distance_to_depot = torch.sum(distance_matrix[:, 0])
    heuristics = heuristics / total_distance_to_depot
    
    # Incorporate exploration by adding some noise to the heuristics
    exploration_noise = torch.rand_like(heuristics) * 0.01
    heuristics += exploration_noise
    
    return heuristics
