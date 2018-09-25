import numpy as np
import csv
import matplotlib.pyplot as plt

def file_reader():
    file = open(r"k.csv")
    reader = csv.reader(file)
    data = list(reader)
    file.close()
    return data

k = int(input('enter number of clusters: ') )# number of clusters

fig = plt.figure()
data = np.asarray(data, dtype=float)

# print(type(data))
# print (data)

def kmeans(data, k):

    centroids = []
    centroids = randomize_centroids(data, centroids, k)

    old_centroids = [[] for i in range(k)]
    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1
            # print('oldcenroids' + str(old_centroids))
    centroids = np.asarray(centroids,dtype=float)
    print("The total number of data instances is: " + str(len(data)))
    print("The total number of iterations necessary is: " + str(iterations))
    print("The means of each cluster are: " + str(centroids))
    # cent_x = centroids[:,0]
    # cent_y = centroids[:,1]
    print(type(centroids))
    cent_x = centroids [:,0]
    cent_y = centroids [:,1]
    plt.scatter(cent_x,cent_y,s=100,marker='x',c='r')
    colors = ['red', 'green', 'blue', 'yellow']
    print("\n The clusters are as follows:\n")

    # print("Cluster_VATIABLE"+str(cluster))
    # print("ClusterS_Variable"+str(clusters))
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y)
    for cluster in clusters:
        print(cluster)
        # print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        # print(np.array(cluster).tolist())

    plt.show()
    print("Cluster ends here.")
    return

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.
def euclidean_dist(data, centroids, clusters):
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        # print('instance',instance)
        # print('centroids',centroids)
        mu_index = min([(i[0], np.linalg.norm(instance-centroids[i[0]])) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
        # print('CENTROIDSSSSSS',centroids)
    return centroids


# check if clusters have converged
def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

kmeans(data,k)

