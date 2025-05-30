import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands are normalized by the total vehicle capacity
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate the sum of demands for each edge
    edge_demands = torch.matmul(normalized_demands, distance_matrix)

    # Calculate the sum of demands for each node (including the depot)
    node_demands = torch.cumsum(edge_demands, dim=1)

    # Create a mask for nodes that have exceeded their capacity
    capacity_exceeded = node_demands > 1.0

    # Create a mask for nodes that have not exceeded their capacity
    capacity_not_exceeded = ~capacity_exceeded

    # For capacity-exceeded nodes, we want to encourage leaving them
    # We do this by penalizing the edges that lead to these nodes
    penalties = -torch.where(capacity_exceeded, torch.ones_like(edge_demands), torch.zeros_like(edge_demands))

    # For capacity-not-exceeded nodes, we want to encourage visiting them
    # We do this by rewarding the edges that lead to these nodes
    rewards = torch.where(capacity_not_exceeded, torch.ones_like(edge_demands), torch.zeros_like(edge_demands))

    # Combine penalties and rewards to form the heuristic matrix
    heuristics = penalties + rewards

    return heuristics