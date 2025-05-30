import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure the distance matrix is symmetric by filling the lower triangle with the upper triangle values
    distance_matrix = distance_matrix + distance_matrix.t() - torch.diag(torch.diag(distance_matrix))

    # Calculate the maximum possible demand per edge as the minimum of the demand of the two nodes
    max_demand_per_edge = torch.min(demands[torch.arange(distance_matrix.shape[0]), torch.arange(distance_matrix.shape[0])],
                                   demands[torch.arange(distance_matrix.shape[0]), torch.arange(distance_matrix.shape[0])].t())

    # Calculate the total demand from the starting node to the rest
    total_demand_from_depot = torch.sum(demands[1:])

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # Loop through all nodes
    for i in range(distance_matrix.shape[0]):
        # Calculate the potential profit of visiting the next node from node i
        # This is done by subtracting the node's demand from the maximum demand
        potential_profit = max_demand_per_edge - demands[i]

        # Update the heuristics matrix
        heuristics[i, 1:] = torch.max(heuristics[i, 1:], potential_profit)
        heuristics[1:, i] = heuristics[i, 1:].clone().t()

    # Add the heuristics from the depot to the rest of the nodes
    heuristics[0, 1:] = torch.max(heuristics[0, 1:], -total_demand_from_depot)

    # Return the heuristics matrix
    return heuristics