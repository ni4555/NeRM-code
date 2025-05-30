import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Assuming that distance_matrix has shape (n, n) and demands has shape (n,)
    # The depot node is indexed by 0, so the demands at the depot are 0.
    # Normalize the distance matrix by dividing by the maximum demand to scale the heuristic.
    max_demand = demands.max()
    normalized_distance_matrix = distance_matrix / max_demand
    
    # The heuristic function is a simple negative of the distance to the depot,
    # with the possibility of adding demand-related information if needed.
    # For simplicity, we use only distance-based heuristics in this example.
    heuristics = -normalized_distance_matrix
    
    # Adjust the heuristics based on customer demands.
    # This could be a simple multiplicative factor that takes into account the demand.
    # Here we use a linearly scaled factor of 1 + (demand/sum_of_demands).
    # This assumes that the demand is normalized already (as per problem description).
    demand_adjustment = (1 + (demands / demands.sum()))
    heuristics *= demand_adjustment
    
    # Ensure that all edges leading to the depot have a heuristic value of 0.
    heuristics[:, 0] = 0
    heuristics[0, :] = 0
    
    return heuristics