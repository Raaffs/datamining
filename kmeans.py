import random

def euclidean_distance(p1,p2):
  return sum((a-b)**2 for a,b in zip(p1,p2))**0.5

def sum_columns(points):
  return [sum(col) for col in zip(*points)] if points else []

def average_columns(sums,count):
  return [ sum / count for sum in sums]

def mean(points):
  return average_columns(sum_columns(points),len(points)) if points else []

def k_means(data,k,max_iters=200):
  centroids = random.sample(data,k)

  for _ in range(max_iters):
    clusters = [[] for _ in range(k)] 
    for point in data:
      cluster_index = min(range(k),key=lambda i: euclidean_distance(point,centroids[i]))
      clusters[cluster_index].append(point)
  
    new_centroids = [mean(cluster) for cluster in clusters]
    if new_centroids == centroids:
      break
    centroids = new_centroids
  return centroids,clusters

data = [
    [1, 2], [2, 1], [4, 5], [5, 4], [8, 9], [9, 8]
]

centroids, clusters = k_means(data, 2)

print("Centroids:", centroids)
print("Clusters:", clusters)