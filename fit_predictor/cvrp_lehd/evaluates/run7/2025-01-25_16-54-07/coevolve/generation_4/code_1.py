import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize a tensor with the same shape as distance_matrix, filled with zeros.
    heuristics = torch.zeros_like(distance_matrix)

    # Calculate the cumulative demand at each node by accumulating the demands from the depot.
    # Since the demands are normalized by the total vehicle capacity, we can use them directly.
    cumulative_demand = demands.cumsum()

    # Nearest neighbor heuristic: Assign each customer to the nearest depot (node 0)
    # by minimizing the distance. This is done by setting the heuristics value to the
    # negative distance from each customer to the depot.
    heuristics[1:, 0] = -distance_matrix[1:, 0]

    # Calculate the cumulative demand check heuristic: Add the negative cumulative demand
    # to the heuristic value for each edge. This encourages avoiding routes that lead to
    # exceeding the vehicle's capacity.
    heuristics += cumulative_demand[1:].unsqueeze(1) + cumulative_demand.unsqueeze(0)

    # Optimize routes based on real-time demand fluctuations: Subtract the distance for
    # each edge to encourage routes with shorter distances. This step is optional and can
    # be adjusted based on the problem's specific needs.
    heuristics -= distance_matrix

    return heuristics