import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Get the number of nodes from the distance matrix
    num_nodes = distance_matrix.shape[0]

    # Calculate the total demand
    total_demand = demands.sum()

    # Calculate the average demand per node
    average_demand = total_demand / num_nodes

    # Calculate the demand per node minus the average demand
    demand_deviation = demands - average_demand

    # Calculate the squared demand deviation
    squared_demand_deviation = demand_deviation ** 2

    # Calculate the heuristic value for each edge
    # Promising edges are those with high deviation from average demand
    # Edges with negative values are undesirable
    heuristic_values = -squared_demand_deviation

    # Subtract the total demand to ensure that the sum of heuristics is equal to the total demand
    # This is to make the heuristic a measure of potential excess capacity that can be used in the heuristic
    heuristic_values -= total_demand

    # Return the heuristic values as a tensor of the same shape as the distance matrix
    return heuristic_values