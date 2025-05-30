import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    depot_index = 0
    demands = demands / demands.sum()  # Normalize demands by total capacity

    # Initialize a tensor with zeros to hold the heuristic values
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Calculate the cost to serve each customer from the depot
    cost_to_depot = distance_matrix[depot_index, 1:]

    # Calculate the heuristic value for each edge ( depot to customer and customer to customer )
    # Promising edges are those that help in load balancing and minimizing distance
    # Here, we use a simple heuristic that considers the demand of the customer
    # and the distance from the depot to the customer
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # Calculate heuristic value for edge from i to j
                edge_heuristic = demands[i] + demands[j] - cost_to_depot[i] - distance_matrix[i, j]
                heuristic_matrix[i, j] = edge_heuristic

    return heuristic_matrix