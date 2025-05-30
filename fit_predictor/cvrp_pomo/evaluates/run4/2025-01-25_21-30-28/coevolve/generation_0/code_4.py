import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands have a shape compatible with the distance_matrix (excluding the depot node)
    demands = demands[1:]

    # Find edges that are not between the depot or two customers that are the same
    is_valid_edge = torch.triu(torch.ones_like(distance_matrix), k=1).to(torch.bool) & \
                    torch.triu(torch.ones_like(distance_matrix), k=0).to(torch.bool)

    # Calculate the sum of demands along each valid edge
    demand_sum = torch.sum(demands[:, None] * demands[None, :], dim=0)

    # Calculate remaining capacity if we were to visit all nodes along the edge
    remaining_capacity = demands.max(dim=0)[0] - demand_sum

    # Calculate the relative prominence of each edge
    prominence = remaining_capacity / demands.max()

    # Scale the prominence by the maximum distance to ensure it's comparable across all edges
    max_distance = distance_matrix.max()
    heuristics = max_distance * prominence

    # Invert the sign to get negative values for undesirable edges
    heuristics = -heuristics

    return heuristics

# Example usage (you will need a distance matrix and demands to test the function)
# distance_matrix = torch.tensor([[0, 5, 8], [5, 0, 7], [8, 7, 0]])
# demands = torch.tensor([1, 3, 2])
# heuristics_matrix = heuristics_v2(distance_matrix, demands)