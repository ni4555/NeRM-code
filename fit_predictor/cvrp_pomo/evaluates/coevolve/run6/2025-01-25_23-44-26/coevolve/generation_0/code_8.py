import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure that the distance matrix and demands are both of type torch.Tensor
    distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
    demands = torch.tensor(demands, dtype=torch.float32)

    # Calculate the cumulative sum of demands, excluding the depot
    cumsum_demands = torch.cumsum(demands[1:], dim=0)

    # Calculate the cumulative sum of distances from the depot to each customer
    cumsum_distances = torch.cumsum(distance_matrix[:, 1:], dim=1)

    # Create a mask where the cumulative demand exceeds the capacity (which is 1 in this case)
    demand_exceeds_capacity = cumsum_demands > 1

    # Initialize the heuristics matrix with zeros
    heuristics = torch.zeros_like(distance_matrix)

    # For each edge (i, j), if the cumulative demand at customer j exceeds capacity,
    # assign a negative value to the edge
    heuristics[:, 1:] = -torch.where(demand_exceeds_capacity, cumsum_distances[:, :-1], 0)

    # For the return to the depot (last column), if the cumulative demand at the last customer exceeds capacity,
    # assign a negative value to the edge
    heuristics[:, -1] = -torch.where(demand_exceeds_capacity[-1:], cumsum_distances[:, -1], 0)

    return heuristics