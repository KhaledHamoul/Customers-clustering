import pandas as pd
import numpy as np
import requests
import json
import operator
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px


# import csv file
def loadCsvFile(path, portion=False):
    print("\n# Importing dataset csv file ...")

    if portion:
        csvFileDf = pd.read_csv('./portion.csv')
    else:
        csvFileDf = pd.read_csv(path)

    print("# dataset csv file imported successfully")
    return csvFileDf


# rename columns
def renameColumns(df, columns):
    df.rename(columns=columns, inplace=True)


# remove empty rows which contains important features with a lot of missing values
def removeEmptyRows(df, columns):
    # replace empty cells by NaN
    df.replace('', np.nan, inplace=True)
    print("\n# Shape before removing lines")
    print(df.shape)
    df = df.dropna(axis=0, subset=columns)
    print("\n# Shape after removing lines")
    print(df.shape)
    return df


def columns(df):
    print('\n\n# Columns')
    print(df.columns)


def countUniqueIds(df, columnName):
    # checking if IDs are unique
    print('\n\n# Number of dataset\'s unique IDs')
    print(df[columnName].nunique())


def head(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print('\n\n# dataset head')
    print(df.head())


def dataTypes(df):
    print('\n\n# dataset features data types')
    print(df.dtypes)


def countEmptyValues(df):
    numberRows = df.shape[0]
    nullsDf = df.isnull().sum()
    nullsDf = pd.DataFrame(nullsDf)
    nullsDf['%'] = round(nullsDf[0] / numberRows * 100, 1)
    nullsDf.rename(columns={0: 'count_missing_values'}, inplace=True)
    print('\n\n# empty/missing values for each feature')
    print(nullsDf)


def countCategoricalValues(df, column):
    print('\n\n# count values for categorical feature : ' + column)
    print(df[column].value_counts(dropna=False, sort=True, normalize=True))


def getMeans(df):
    print('\n\n# features\' means')
    # print(df.slaapkamers.mean())
    # print(df.aantal_personen.mean())


def impute(df, column, value):
    df[column].fillna(value, inplace=True)


def simpleOneHot(df, column, prefix):
    newColumns = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, newColumns], axis=1)
    del df[column]
    return df


def multipleChoicesOneHot(df, column, prefix, categories, state=False):
    if state:
        file = open('states.json', 'r+')
        data = json.load(file)
        for oldCategoryName, newCategoryName in categories.items():
            newRegionColumn = []
            for item in df[column]:
                addresses = item.split(',') if type(item) is str else []
                if addresses:
                    foundCount = 0
                    for address in addresses:
                        address = address.strip()
                        if address != '':
                            state = getState(address, file, data)
                            if newCategoryName == state:
                                foundCount += 1
                    newRegionColumn.append(foundCount)
                else:
                    newRegionColumn.append(np.nan)

            df[prefix + '_' + newCategoryName] = newRegionColumn
    else:
        for oldCategoryName, newCategoryName in categories.items():
            df[prefix + '_' + newCategoryName] = df[column].str.contains(oldCategoryName)

        for oldCategoryName, newCategoryName in categories.items():
            df.loc[df[prefix + '_' + newCategoryName] == True, prefix + '_' + newCategoryName] = 1
            df.loc[df[prefix + '_' + newCategoryName] == False, prefix + '_' + newCategoryName] = 0

    del df[column]


def getState(address, file, data):
    if address == 'outside':
        return 'outside'

    state = ''
    if address in data:
        state = data[address]
    else:
        apiUrl = 'https://nominatim.openstreetmap.org/search?q=' + address + '+nederland&format=json&polygon_geojson=1&addressdetails=1'
        response = requests.get(apiUrl)
        if response.status_code == 200:
            try:
                state = response.json()[0]['address']['state']
                data.update({address: state})
                file.seek(0)
                json.dump(data, file)
            except Exception:
                print(address + ' not found!')
        else:
            return Exception('Api call error!')

    return state


def minMaxBudget(df):
    maxBuyingBudgets = []
    minBuyingBudgets = []

    maxRentingBudgets = []
    minRentingBudgets = []

    minBudgets = []
    maxBudgets = []

    for index, item in df.iterrows():
        column = 'buying_budget' if df.buy[index] == 1 else 'renting_budget'
        if df[column][index] is not np.nan and df[column][index] != '':
            budgets = df[column][index].replace('-', ',').split(',') if type(df[column][index]) is str else []
            budgets = [x for x in budgets if x]
            for key in range(0, len(budgets)):
                if budgets[key] == 'onder 150000': budgets[key] = '100000'
                if budgets[key] == 'boven 1000000': budgets[key] = '2000000'
                if budgets[key] == 'onder 500': budgets[key] = '250'
                if budgets[key] == 'boven 2000': budgets[key] = '4000'

            maxBudget = int(max(budgets, key=lambda item: int(item)))
            minBudget = int(min(budgets, key=lambda item: int(item)))

            maxBudgets.append(maxBudget)
            minBudgets.append(minBudget)

            if column == 'buying_budget':
                maxBuyingBudgets.append(maxBudget)
                minBuyingBudgets.append(minBudget)
            else:
                maxRentingBudgets.append(maxBudget)
                minRentingBudgets.append(minBudget)

        else:
            maxBudgets.append('nan_' + column)
            minBudgets.append('nan_' + column)

    df['max_budget'] = maxBudgets
    df['min_budget'] = minBudgets

    medians = {}
    # buying_budgets medians
    medians['max_buying_median'] = np.median(maxBuyingBudgets)
    medians['min_buying_median'] = np.median(minBuyingBudgets)

    # renting_budgets medians
    medians['max_renting_median'] = np.median(maxRentingBudgets)
    medians['min_renting_median'] = np.median(minRentingBudgets)

    del df['buying_budget']
    del df['renting_budget']

    return medians


# Data training helpers
# https://stackoverflow.com/questions/48036593/is-my-python-implementation-of-the-davies-bouldin-index-correct
def daviesBouldin(X, labels):

    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis=0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []

    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

    return np.max(db) / n_cluster

# Dunn index for clustering
# https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6 [old]
def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]
    return np.min(values)


def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    return np.max(values)

def dunn(datasetDf, labels):
    from sklearn.metrics.pairwise import euclidean_distances

    distances = euclidean_distances(datasetDf)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di

def silhouetteScore(datasetDf, num_clusters):
    import matplotlib.cm as cm
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.cluster import KMeans

    silhouetteAvgs = {}
    avgs = []

    range_n_clusters = range(2, num_clusters + 1)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(18, 7)

    for n_clusters in range_n_clusters:
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(datasetDf) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(datasetDf)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(datasetDf, cluster_labels)

        silhouetteAvgs[n_clusters] = silhouette_avg
        avgs.append(silhouette_avg)

        if num_clusters == n_clusters:
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                datasetDf, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(np.array(datasetDf)[:, 0], np.array(datasetDf)[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

    ax3.set_title("The Silhouette Method avgs")
    ax3.set_xlabel("Cluster label")
    ax3.set_ylabel("The silhouette coefficient values")
    ax3.plot(range_n_clusters, avgs, 'bx-')

    plt.show()


def plotPca3d(datasetDf, labels=None, sample_points=None, matplt=False):
    plt.style.use('seaborn-whitegrid')

    if labels is None:
        labels = [1 for i in range(datasetDf.shape[0])]

    pca = PCA(n_components=3, svd_solver='full')
    pcaTempDataset = pca.fit_transform(datasetDf)

    pcaTempDataset = np.append(pcaTempDataset, labels.reshape(-1, 1), axis=1)

    if sample_points is None:
        reducedDataset = pcaTempDataset
    else:
        reducedDataset = pcaTempDataset[np.random.choice(pcaTempDataset.shape[0], sample_points, replace=False)]

    if matplt:
        fig = plt.figure(1, figsize=(100, 60))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()

        ax.scatter(reducedDataset[:, 0], reducedDataset[:, 1], reducedDataset[:, 2], c=reducedDataset[:, 3], cmap=plt.cm.nipy_spectral, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        plt.show()
    else:
        fig = px.scatter_3d(
            reducedDataset, x=0, y=1, z=2, color=reducedDataset[:, 3],
            title=f'Total Explained Variance (PCA): ',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()

def tsne(datasetDf, labels, sample_points=None):
    from sklearn.manifold import TSNE

    perplexity = 10
    print('before tsne')
    tsne = TSNE(n_components=3, perplexity=perplexity)
    tsneTempDf = pd.DataFrame(tsne.fit_transform(datasetDf))

    print('after tsne')

    tsneTempDf = np.append(tsneTempDf, labels.reshape(-1, 1), axis=1)

    if sample_points is None:
        reducedDataset = tsneTempDf
    else:
        reducedDataset = tsneTempDf[np.random.choice(tsneTempDf.shape[0], sample_points, replace=False)]

    print('reduced tsne df')
    fig = px.scatter_3d(
        reducedDataset, x=0, y=1, z=2, color=reducedDataset[:, 3],
        title=f'Total Explained Variance (T-SNE): ',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()

def parallelCoordinates(datasetDf, labels, num_clusters):
    from pandas.plotting import parallel_coordinates
    import seaborn as sns

    palette = sns.color_palette("bright", 10)

    X_clustered = pd.DataFrame(datasetDf)
    X_clustered["cluster"] = labels

    df = X_clustered

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster == i])

    # Create the plot
    fig = plt.figure(figsize=(24, 30))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):
        plt.subplot(num_clusters, 1, i + 1)
        for j, c in enumerate(cluster_points):
            if i != j:
                pc = parallel_coordinates(c, 'cluster', color=[(palette[j][0], palette[j][1], palette[j][2], 0.2)])

        pc = parallel_coordinates(cluster_points[i], 'cluster',
                                  color=[(palette[i][0], palette[i][1], palette[i][2], 0.5)])

        # Stagger the axes
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)

    plt.show()


