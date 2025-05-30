import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    max_demand = demands.max()
    min_demand = demands.min()
    demand_span = max_demand - min_demand

    # Normalize the demands to the range [0, 1]
    normalized_demands = (demands - min_demand) / demand_span

    # Node partitioning: create partitions based on normalized demand
    demand_thresholds = torch.linspace(0, 1, steps=n // 2 + 1, dtype=torch.float32)
    partitions = torch.zeros(n, dtype=torch.long)
    for i, threshold in enumerate(demand_thresholds):
        partitions[torch.where(normalized_demands < threshold)] = i

    # Demand relaxation: increase demand within the same partition
    relaxed_demands = torch.zeros_like(demands)
    for i in range(n // 2 + 1):
        partition = partitions[partitions == i]
        relaxed_demand_sum = demands[partition].sum()
        relaxed_demand_avg = relaxed_demand_sum / partition.numel()
        relaxed_demands[partition] = relaxed_demand_avg

    # Path decomposition: create a new matrix where each edge's heuristics are based on
    # the relaxed demands of its nodes
    heuristics_matrix = -torch.abs(distance_matrix - relaxed_demands.unsqueeze(1))

    # Dynamic window approach: adjust heuristics based on vehicle capacities
    vehicle_capacities = torch.ones(n, dtype=torch.float32)  # Assume all vehicles have the same capacity
    heuristics_matrix = heuristics_matrix + (vehicle_capacities - relaxed_demands).unsqueeze(1)

    # Multi-objective evolutionary algorithm: promote edges with lower distance and better load balance
    load_balance = (demands.sum() - max_demand) / demands.sum()
    heuristics_matrix = heuristics_matrix + (1 - load_balance) * distance_matrix

    return heuristics_matrix