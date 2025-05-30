import numpy as np
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Initialize the probability distribution
    probabilities = np.ones(n) / n
    
    # KMeans clustering to find initial clusters
    kmeans = KMeans(n_clusters=m, random_state=0).fit(weight)
    
    # Calculate the entropy of the clusters
    cluster_entropy = entropy(kmeans.labels_, probabilities)
    
    # Iterative optimization
    for _ in range(100):
        # Update probabilities based on the prize and weight
        probabilities = (prize * (1 - weight)).sum(axis=1) / (1 - weight).sum(axis=1)
        
        # KMeans clustering with updated probabilities
        kmeans.fit(weight)
        
        # Calculate the entropy of the clusters
        new_cluster_entropy = entropy(kmeans.labels_, probabilities)
        
        # If entropy decreases, continue; otherwise, adjust probabilities
        if new_cluster_entropy < cluster_entropy:
            cluster_entropy = new_cluster_entropy
        else:
            probabilities = np.ones(n) / n
    
    # Calculate the heuristics based on the final probabilities
    heuristics = probabilities * (prize * (1 - weight)).sum(axis=1)
    
    return heuristics
