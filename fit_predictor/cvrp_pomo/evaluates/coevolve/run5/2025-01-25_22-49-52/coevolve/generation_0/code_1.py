import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # The total demand per vehicle can be a priori determined by the maximum of demands or based on an assumption
    # For simplicity, we'll use max of the demands to denote vehicle capacity in terms of demands.
    vehicle_capacity = torch.max(demands)

    # Compute a score that reflects how desirable it is to visit a customer.
    # The heuristic for an edge is negative for edges where visiting a customer makes it too likely to exceed vehicle capacity.
    # A basic heuristic might be that a customer with a high normalized demand should be visited earlier rather than later.
    # The heuristic can be calculated as follows:
    # (1 - (customer_demand / vehicle_capacity)) * distance_matrix - excess_demand_score

    # Compute excess_demand_score where a customer would exceed the vehicle capacity
    # if visited before being fully utilized. We subtract this score to penalize those edges.
    excess_demand_score = demands * (demands / vehicle_capacity) * distance_matrix

    # Now compute the positive part of the heuristic:
    # Subtract from the base cost to make a smaller value a higher priority
    heuristic_score = (vehicle_capacity - demands) / vehicle_capacity
    heuristic_score = (1 - heuristic_score) * distance_matrix

    # The total heuristic value for an edge (i, j) would be:
    # negative score for when customer at node i's demand, when added to what has already been visited, would exceed vehicle capacity
    # and a positive score when considering just the cost and the remaining demand.

    # Combine the negative excess_demand_score with the positive heuristic_score.
    heuristic_matrix = heuristic_score - excess_demand_score

    return heuristic_matrix

# Example usage:
# n is the number of nodes
# Assuming distance_matrix is given in the form of a torch.Tensor
# demands is given in the form of a torch.Tensor where each element represents the demand of a node

# n = 5
# distance_matrix = torch.tensor([
#     [0, 5, 6, 4, 1],
#     [5, 0, 1, 9, 4],
#     [6, 1, 0, 5, 6],
#     [4, 9, 5, 0, 7],
#     [1, 4, 6, 7, 0]
# ], dtype=torch.float)
# demands = torch.tensor([1.0, 2.0, 2.5, 2.5, 2.5], dtype=torch.float)
# result = heuristics_v2(distance_matrix, demands)