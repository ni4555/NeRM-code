import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the same shape as distance_matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the shortest path heuristics using a simple Dijkstra's algorithm
    for start in range(distance_matrix.shape[0]):
        # Create a priority queue to select the next node with the shortest distance
        priority_queue = [(0, start)]
        # Initialize distances with infinity
        distances = np.full(distance_matrix.shape, np.inf)
        distances[start] = 0
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # If the current node's distance is already greater than the shortest path found,
            # there is no need to continue from this node
            if current_distance > distances[current_node]:
                continue
            
            # Update the distances for each neighboring node
            for neighbor in range(distance_matrix.shape[1]):
                if distance_matrix[current_node, neighbor] > 0:
                    new_distance = current_distance + distance_matrix[current_node, neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor))
        
        # Set the heuristics array to the shortest distances from the start node
        heuristics[start] = distances
    
    return heuristics