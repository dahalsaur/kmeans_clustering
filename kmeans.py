import numpy as np
from matplotlib import pyplot as plt
import random

        #kmeans algorithm
def kmeans(dataset, k):
            #array to store the cluster index(1....k) of every class
    class_cluster_index = []
    for i in range(len(dataset)):
        class_cluster_index.append(random.randint(1,k))
        #a copy of a class_cluster_index to compare the changes
    old_class_cluster_index = class_cluster_index
    while (1):       #infinite loop until there is no changes in the cluster index of the classes
            #array to store k centroids
        centroids = [[0]*len(dataset[0])]*k
        sample = [0]*len(dataset[0])
            #finding centroids for each clusters
        j=0

        for l in class_cluster_index:
            if list(centroids[l-1]) == sample:
                centroids[l-1] = dataset[j]
            else:
                        #centroids is YET TO BE FIXED(Implement your own mean algorithm)
                new = np.array([centroids[l-1], dataset[j]])
                centroids[l-1] = new.sum(axis=0)
                centroids[l-1] = np.floor(centroids[l-1])
            j += 1
        centroids = (np.array(centroids))/2

                #finding nearest image vector from the centroid using eucledian distance
        for i in range(len(dataset)):
            euc_dist = 1000
            for j in range(k):
                dist = np.linalg.norm(dataset[i]-centroids[j])
                if dist < euc_dist:
                    class_cluster_index[i] = j+1
                    euc_dist = dist
        if (old_class_cluster_index == class_cluster_index):
            return class_cluster_index
        else:
            old_class_cluster_index = class_cluster_index
        print(centroids)

    return centroids


    #Importing the dataset
#full_dataset = np.loadtxt("mfeat-pix.txt")
#dataset = full_dataset[200:400]
dataset = [[1,2,3],[4,5,6],[7,8,9]]
k = 2
centroids = kmeans(dataset, k)
print(centroids)
        #plot the k clusters of size 16*15
for i in range (0,k):
    cluster_i = np.resize(centroids[i],(16,15))
    plt.imshow(cluster_i, cmap = 'gray')
    plt.show()
