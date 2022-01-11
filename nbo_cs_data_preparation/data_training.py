import helpers
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

if __name__ == '__main__':
    print("== Script Started ==")

    sample_points = 100
    num_clusters = 3

    datasetDf = pd.read_csv('./nbo_dataset.csv')
    print(datasetDf.shape)

    algoResult = KMeans(n_clusters=num_clusters, init="random", random_state=1).fit(datasetDf)

    labels = algoResult.labels_
    # distance = euclidean_distances(datasetDf)

    print('\nDunn indexes')
    di = helpers.dunn()
    print(di)

    print('\nDavies bouldin score')
    dbs = davies_bouldin_score(datasetDf, labels)
    print(dbs)

    print('\nInertia')
    inertia = algoResult.inertia_
    print(inertia)

    print('\nPlot 3D')
    helpers.plotPca3d(datasetDf=datasetDf, labels=labels, sample_points=sample_points, matplt=False)

    print('\nSilhouette score')
    helpers.silhouetteScore(datasetDf, num_clusters=num_clusters)

    print('\nParallel coordinates')
    helpers.parallelCoordinates(datasetDf=datasetDf, labels=labels, num_clusters=num_clusters)

    print('\nT-SNE ploting')
    helpers.tsne(datasetDf=datasetDf, sample_points=sample_points, labels=labels)




