import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the demands vector has a sum close to 1 to normalize the vehicle capacity
    demands = demands / demands.sum()

    # Compute the cumulative demand along each row of the distance matrix
    cumulative_demands = torch.cumsum(demands, dim=1)

    # Calculate the potential reward for each edge as the negative of the distance
    # multiplied by the cumulative demand. This encourages closer nodes to be visited first.
    reward = -distance_matrix * cumulative_demands

    # To avoid including edges that have zero or negative potential reward, we can add a
    # small constant (e.g., 1e-5) to the reward values before exponentiating.
    # This will help to ensure that no edge is penalized too heavily due to the negative
    # distance values.
    reward = reward + 1e-5

    # Exponentiate the reward to create a heuristic that gives higher weights to promising edges
    heuristic_matrix = torch.exp(reward)

    return heuristic_matrix