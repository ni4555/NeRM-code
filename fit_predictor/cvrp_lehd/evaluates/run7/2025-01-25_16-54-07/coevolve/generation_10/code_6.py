import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Step 1: Normalize demands to reflect total demand per node
    total_demand = demands.sum()
    normalized_demands = demands / total_demand

    # Step 2: Create a cumulative demand mask for load balancing
    cumulative_demand_mask = torch.cumsum(normalized_demands, dim=0)

    # Step 3: Create an edge feasibility mask for capacity constraint
    vehicle_capacity = demands[0]  # Assuming the depot's demand represents vehicle capacity
    edge_capacity_mask = distance_matrix < vehicle_capacity

    # Step 4: Evaluate edges based on demand and capacity
    # Combine the cumulative demand with edge capacity to evaluate edges
    edge_evaluation = (cumulative_demand_mask * edge_capacity_mask) * (distance_matrix != 0)

    # Step 5: Prioritize edges by their evaluation
    # Higher positive values indicate more promising edges
    # We use negative values to indicate undesirable edges
    # We can set a threshold to differentiate between promising and undesirable edges
    threshold = -0.5  # This threshold can be adjusted based on problem specifics
    promising_edges = edge_evaluation - threshold
    undesirable_edges = edge_evaluation + threshold

    # Replace undesirable edges with very negative values to ensure they are not chosen
    edge_priors = torch.where(undesirable_edges > 0, undesirable_edges, promising_edges)

    return edge_priors