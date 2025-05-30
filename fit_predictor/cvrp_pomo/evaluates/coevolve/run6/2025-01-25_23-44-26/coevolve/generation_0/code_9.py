import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the sum of the demands of all nodes except the depot
    non_depot_demand_sum = demands[1:].sum()

    # Calculate the negative of the distance to the depot for each node (to be used as a heuristic)
    negative_depot_distances = -distance_matrix[:, 0]

    # Calculate the potential profit of visiting a node, which is the sum of the negative distance
    # to the depot and the normalized demand of the node (assuming higher demand is better)
    potential_profit = negative_depot_distances + demands[1:]

    # Normalize the potential profit by the total vehicle capacity
    # Here, we are assuming the demands are already normalized by the total capacity
    normalized_profit = potential_profit / non_depot_demand_sum

    # The result should be a matrix where the value at index [i, j] indicates how promising it is
    # to include the edge from node i to node j in a solution. We use a threshold to ensure
    # negative values for undesirable edges and positive values for promising ones.
    # For this example, we use a simple threshold, but in practice, this might be a parameter.
    threshold = 0.5
    result = torch.where(normalized_profit > threshold, normalized_profit, -torch.inf)

    return result