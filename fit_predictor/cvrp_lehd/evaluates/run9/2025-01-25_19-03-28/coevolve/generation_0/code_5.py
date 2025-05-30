import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum possible demand for each node
    max_demand = demands / demands.sum()
    
    # Calculate the potential benefit of each edge
    # Promising edges are those where the maximum demand of the customer is less than the distance to the next customer
    # We use a negative value for undesirable edges (large distance or high demand)
    edge_benefit = (distance_matrix[1:] < distance_matrix[:-1, 1:]).float() * (1 - max_demand[:-1])
    
    # Incorporate customer demand into the edge benefit
    edge_benefit = edge_benefit * demands[1:]
    
    # For the depot to customer edges, the benefit is the demand
    edge_benefit[0, 1:] = demands[1:]
    
    # For the customer to depot edge, the benefit is the negative of the demand
    edge_benefit[1:, 0] = -demands[1:]
    
    return edge_benefit