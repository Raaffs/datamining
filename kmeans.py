import random
import matplotlib.pyplot as plt

# Generate random data points
data = [
    [random.randint(0, 150), random.randint(0, 150)] for _ in range(1000)
]

def euclidean_distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def sum_columns(points):
    return [sum(col) for col in zip(*points)] if points else []

def average_columns(sums, count):
    return [sum / count for sum in sums]

def mean(points):
    return average_columns(sum_columns(points), len(points)) if points else []

def k_means(data, k, max_iters=200):
    centroids = random.sample(data, k)

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in data:
            cluster_index = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[cluster_index].append(point)

        new_centroids = [mean(cluster) for cluster in clusters]
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids, clusters

# Apply K-Means to the data
centroids, clusters = k_means(data, 3)  # You can change k to whatever value you want

# Plotting the clusters
plt.figure(figsize=(8, 6))  # Set the figure size
for i, cluster in enumerate(clusters):
    cluster_points = list(zip(*cluster))
    plt.scatter(cluster_points[0], cluster_points[1], label=f"Cluster {i + 1}", s=100, alpha=0.6, edgecolors='w')

centroid_x, centroid_y = zip(*centroids)
plt.scatter(centroid_x, centroid_y, marker='x', color='black', s=200, label="Centroids", linewidths=2)

# Aesthetic enhancements
plt.title("K-Means Clustering", fontsize=16, weight='bold')
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
plt.legend()
plt.show()
