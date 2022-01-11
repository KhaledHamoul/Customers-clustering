import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import helpers
import operator
import json
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

if __name__ == '__main__':
    print("== Script Started ==")

    # laod csv file
    csvFileDf = helpers.loadCsvFile('./housing_needs_with_region_2020-08-14_03-49-24.csv', portion=False)

    # count unique IDs
    helpers.countUniqueIds(csvFileDf, columnName='id')

    ids = csvFileDf['id']
    del csvFileDf['id']

    # rename columns to english words
    helpers.renameColumns(csvFileDf, {
        'slaapkamers': 'bedrooms',
        'leeftijd': 'age_category',
        'gezinssamenstelling': 'familly_composition',
        'slaapkamer_begane_grond': 'bedroom_ground_floor',
        'badkamers': 'bathrooms',
        'aantal_personen': 'moving_persons',
        'verhuistermijn': 'moving_period',
        'type_huidige_woning': 'current_home_type',
        'woonoppervlakte': 'living_space',
        'perceeloppervlakte': 'property_size',
        'koophuur': 'buy_rent_choice',
        'budget': 'buying_budget',
        'budgethuur': 'renting_budget',
        'buitenruimte': 'areas_outdoors',
        'daktypes': 'roof_type',
        'uitzicht': 'living_room_view',
        'woningstijl': 'building_style',
        'woningtype': 'building_type'
    })

    helpers.countEmptyValues(csvFileDf)

    del csvFileDf['postalcode']
    del csvFileDf['bezit_vrijekavel']
    del csvFileDf['roof_type']
    del csvFileDf['building_style']

    # remove empty lines for important features
    csvFileDf = helpers.removeEmptyRows(csvFileDf, columns=['age_category', 'familly_composition', 'moving_persons', 'current_home_type'])

    # display columns names
    helpers.columns(csvFileDf)

    # display the head of the dataset
    helpers.head(csvFileDf)

    # display data type of the features
    helpers.dataTypes(csvFileDf)

    # count missing values for each feature
    helpers.countEmptyValues(csvFileDf)

    # get the means for numerical features
    helpers.getMeans(csvFileDf)

    # impute "bedrooms"
    helpers.countCategoricalValues(csvFileDf, 'bedrooms')
    helpers.impute(csvFileDf, 'bedrooms', 2.5)  # 3 => 35% && 2 => 33%
    helpers.countCategoricalValues(csvFileDf, 'bedrooms')

    # age_category
    csvFileDf.loc[csvFileDf['age_category'] == 'onder30', 'age_category'] = 25
    csvFileDf.loc[csvFileDf['age_category'] == '30-45', 'age_category'] = 37
    csvFileDf.loc[csvFileDf['age_category'] == '46-55', 'age_category'] = 50
    csvFileDf.loc[csvFileDf['age_category'] == '56-65', 'age_category'] = 60
    csvFileDf.loc[csvFileDf['age_category'] == 'boven65', 'age_category'] = 70

    # familly_composition
    csvFileDf = helpers.simpleOneHot(csvFileDf, 'familly_composition', 'familly_composition')
    helpers.renameColumns(csvFileDf, {
        'familly_composition_alleen': 'familly_composition_single',
        'familly_composition_alleen-met': 'familly_composition_single_with_children',
        'familly_composition_samen': 'familly_composition_couple',
        'familly_composition_samen-met': 'familly_composition_couple_with_children'
    })
    print(csvFileDf.shape)
    print(csvFileDf.columns)

    # current_home_type
    csvFileDf = helpers.simpleOneHot(csvFileDf, 'current_home_type', 'current_home_type')
    helpers.renameColumns(csvFileDf, {
        'current_home_type_koop': 'current_home_type_owned',
        'current_home_type_huur': 'current_home_type_rented',
        'current_home_type_anders': 'current_home_type_other'
    })
    print(csvFileDf.shape)
    print(csvFileDf.columns)

    # impute "bedroom_ground_floor"
    helpers.countCategoricalValues(csvFileDf, 'bedroom_ground_floor')
    helpers.impute(csvFileDf, 'bedroom_ground_floor', 'mag')  # mag => 44%
    csvFileDf = helpers.simpleOneHot(csvFileDf, 'bedroom_ground_floor', 'bedroom_ground_floor')
    print(csvFileDf.shape)

    # impute "bathrooms"
    helpers.countCategoricalValues(csvFileDf, 'bathrooms')
    helpers.impute(csvFileDf, 'bathrooms', 1)  # 1 => 87%
    helpers.countCategoricalValues(csvFileDf, 'bathrooms')

    # grouping & impute "moving_period"
    helpers.countCategoricalValues(csvFileDf, 'moving_period')
    csvFileDf.loc[
        (csvFileDf['moving_period'] == 'binnen1') | (csvFileDf['moving_period'] == 'binnen2'), 'moving_period'] = 'lt2'
    helpers.countCategoricalValues(csvFileDf, 'moving_period')
    helpers.impute(csvFileDf, 'moving_period', 'lt2')  # binnen1 => 36% & binnen2 => 32% / total 69%
    helpers.countCategoricalValues(csvFileDf, 'moving_period')
    csvFileDf = helpers.simpleOneHot(csvFileDf, 'moving_period', 'moving_period')  # One-hot
    helpers.renameColumns(csvFileDf, {
        'moving_period_na5': 'moving_period_gt_5',
        'moving_period_onbekend': 'moving_period_unknown',
        'moving_period_tussen2en5': 'moving_period_btw_2_5'
    })
    print(csvFileDf.shape)

    # impute "living_space"
    csvFileDf.loc[csvFileDf['living_space'] == 0, 'living_space'] = 25
    csvFileDf.loc[csvFileDf['living_space'] == 50, 'living_space'] = 62.5
    csvFileDf.loc[csvFileDf['living_space'] == 75, 'living_space'] = 87.5
    csvFileDf.loc[csvFileDf['living_space'] == 100, 'living_space'] = 107.5
    csvFileDf.loc[csvFileDf['living_space'] == 115, 'living_space'] = 122.5
    csvFileDf.loc[csvFileDf['living_space'] == 130, 'living_space'] = 140
    csvFileDf.loc[csvFileDf['living_space'] == 150, 'living_space'] = 200
    csvFileDf.loc[csvFileDf['living_space'] == 250, 'living_space'] = 250
    helpers.countCategoricalValues(csvFileDf, 'living_space')
    helpers.impute(csvFileDf, 'living_space', csvFileDf['living_space'].median())  # median = 107.5 / mean = 110.5
    helpers.countCategoricalValues(csvFileDf, 'living_space')

    # impute "property_size"
    csvFileDf.loc[csvFileDf['property_size'] == 0, 'property_size'] = 50
    csvFileDf.loc[csvFileDf['property_size'] == 100, 'property_size'] = 175
    csvFileDf.loc[csvFileDf['property_size'] == 250, 'property_size'] = 375
    csvFileDf.loc[csvFileDf['property_size'] == 500, 'property_size'] = 750
    csvFileDf.loc[csvFileDf['property_size'] == 1000, 'property_size'] = 1750
    csvFileDf.loc[csvFileDf['property_size'] == 2500, 'property_size'] = 2500
    helpers.countCategoricalValues(csvFileDf, 'property_size')
    helpers.impute(csvFileDf, 'property_size', csvFileDf['property_size'].median())  # median = 175 / mean = 239.6
    helpers.countCategoricalValues(csvFileDf, 'property_size')

    # impute & apply One-hot "buy_rent_choice"
    print(csvFileDf.shape)
    helpers.countCategoricalValues(csvFileDf, 'buy_rent_choice')
    helpers.impute(csvFileDf, 'buy_rent_choice', 'koop')  # impute
    helpers.countCategoricalValues(csvFileDf, 'buy_rent_choice')

    # remove rows where 'buy_rent_choice' is nvt
    print(csvFileDf.shape)
    csvFileDf['buy_rent_choice'].replace('nvt', np.nan, inplace=True)
    csvFileDf = csvFileDf.dropna(axis=0, subset=['buy_rent_choice'])
    print(csvFileDf.shape)

    csvFileDf = helpers.simpleOneHot(csvFileDf, 'buy_rent_choice', '')  # One-hot
    helpers.renameColumns(csvFileDf, {
        '_huur': 'rent',
        '_koop': 'buy'
    })
    print(csvFileDf.columns)

    # renting_budget & buying_budget
    medians = helpers.minMaxBudget(csvFileDf)

    csvFileDf.loc[csvFileDf['max_budget'] == 'nan_buying_budget', 'max_budget'] = medians['max_buying_median']
    csvFileDf.loc[csvFileDf['min_budget'] == 'nan_buying_budget', 'min_budget'] = medians['min_buying_median']
    csvFileDf.loc[csvFileDf['max_budget'] == 'nan_renting_budget', 'max_budget'] = medians['max_renting_median']
    csvFileDf.loc[csvFileDf['min_budget'] == 'nan_renting_budget', 'min_budget'] = medians['min_renting_median']

    print(csvFileDf.columns)

    # impute "areas_outdoors"
    categories = {
        'balkon': 'balcony',
        'dakterras': 'terrace',
        'tuin': 'garden',
        'parkeerplaats': 'park_1',
        'dubbele-parkeerplaats': 'park_2',
        'zonnepanelen': 'solar_panels'
    }
    column = 'areas_outdoors'
    prefix = column + '_'

    helpers.multipleChoicesOneHot(csvFileDf, column=column, prefix=column, categories=categories)
    print(csvFileDf.shape)

    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    for category in categories.values():
        if category == "garden":
            helpers.impute(csvFileDf, prefix + category, 1)  # impute NaN garden values by 1 (frequent)
        else:
            helpers.impute(csvFileDf, prefix + category, 0)  # impute other values by 0

    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    # impute "living_room_view"
    categories = {
        'tuin': 'garden',
        'straat': 'street'
    }
    column = 'living_room_view'
    prefix = column + '_'
    helpers.multipleChoicesOneHot(csvFileDf, column=column, prefix=column, categories=categories)
    print(csvFileDf.shape)

    for category in categories.values():
        helpers.impute(csvFileDf, prefix + category, 0)  # impute values by 0 (not chosen)

    # impute "building_type"
    categories = {
        'vrijstaand': 'detached_house',
        '2onder1kap': 'semi_detached_house',
        'rij': 'terraced_house',
        'appartement': 'apartment',
        'bungalow': 'bangalow'
    }
    column = 'building_type'
    prefix = column + '_'

    helpers.multipleChoicesOneHot(csvFileDf, column=column, prefix=column, categories=categories)
    print(csvFileDf.shape)

    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    for category in categories.values():
        if category in ["terraced_house", "semi_detached_house", "apartment"]:
            helpers.impute(csvFileDf, prefix + category, 1)  # impute NaN by top 3 frequent categories
        else:
            helpers.impute(csvFileDf, prefix + category, 0)  # impute other values by 0

    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())


    # impute "city"
    file = open('states.json', 'r+')
    data = json.load(file)
    states = {}
    for item in csvFileDf['city']:
        addresses = item.split(',') if type(item) is str else []
        for address in addresses:
            address = address.strip()
            if address != '':
                state = helpers.getState(address, file, data)
                if state in states:
                    states[state] += 1
                else:
                    states[state] = 1

    file.close()
    topMunicipalities = sorted(states.items(), key=operator.itemgetter(1), reverse=True)[:12]
    print(len(states))
    print(topMunicipalities)

    categories = {
        'outside': 'outside',
        'Utrecht': 'Utrecht',
        'Groningen': 'Groningen',
        'Zuid-Holland': 'Zuid-Holland',
        'Noord-Holland': 'Noord-Holland',
        'Drenthe': 'Drenthe',
        'Overijssel': 'Overijssel',
        'Gelderland': 'Gelderland',
        'Noord-Brabant': 'Noord-Brabant',
    }
    column = 'city'
    prefix = 'from_region'

    helpers.multipleChoicesOneHot(csvFileDf, column=column, prefix=prefix, categories=categories, state=True)
    print(csvFileDf.shape)

    prefix += '_'
    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    for category in categories.values():
        if category in ["Utrecht"]:
            helpers.impute(csvFileDf, prefix + category, 1)  # impute NaN by top frequent category
        else:
            helpers.impute(csvFileDf, prefix + category, 0)  # impute other values by 0
    print('====================================')
    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    # impute "regions"
    file = open('states.json', 'r+')
    data = json.load(file)
    states = {}
    for item in csvFileDf['region']:
        addresses = item.split(',') if type(item) is str else []
        for address in addresses:
            address = address.strip()
            if address != '':
                state = helpers.getState(address, file, data)
                if state in states:
                    states[state] += 1
                else:
                    states[state] = 1

    file.close()
    topMunicipalities = sorted(states.items(), key=operator.itemgetter(1), reverse=True)[:12]
    print(len(states))
    print(topMunicipalities)

    categories = {
        'Utrecht': 'Utrecht',
        'Groningen': 'Groningen',
        'Zuid-Holland': 'Zuid-Holland',
        'Noord-Holland': 'Noord-Holland',
        'Drenthe': 'Drenthe',
        'Overijssel': 'Overijssel',
        'Gelderland': 'Gelderland',
        'Noord-Brabant': 'Noord-Brabant',
    }
    column = 'region'
    prefix = 'to_region'

    helpers.multipleChoicesOneHot(csvFileDf, column=column, prefix=prefix, categories=categories, state=True)
    print(csvFileDf.shape)

    prefix += '_'
    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    for category in categories.values():
        if category in ["Utrecht", "Zuid-Holland", "Groningen", "Noord-Holland"]:
            helpers.impute(csvFileDf, prefix + category, 1)  # impute NaN by top 3 frequent categories
        else:
            helpers.impute(csvFileDf, prefix + category, 0)  # impute other values by 0
    print('====================================')
    for category in categories.values():
        print(prefix + category + ' : ', csvFileDf[prefix + category].sum())

    scaler = MinMaxScaler()
    scaledDataset = scaler.fit_transform(csvFileDf.values)

    csvFileDf.loc[:, :] = scaledDataset

    # csvFileDf.to_csv(r'.\nbo_dataset.csv', index=False, header=True)

    print('============ Data preparation finished succefully')

    from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, MeanShift

    # kmeans = KMeans(n_clusters=8, algorithm='full').fit(scaledDataset)
    # kmeans = MiniBatchKMeans(n_clusters=4).fit(scaledDataset)
    kmeans = SpectralClustering(n_clusters=4).fit(scaledDataset)
    # kmeans = DBSCAN().fit(scaledDataset)
    # kmeans = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(scaledDataset) # Kaggle
    # kmeans = MeanShift().fit(scaledDataset)

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    pca = PCA(n_components=3)
    pca.fit(scaledDataset)

    print('K-Means started...')
    x_pca = pca.transform(scaledDataset)
    print('clustering is done !!')

    print(scaledDataset.shape)
    print(x_pca.shape)

    # 3D
    fig = plt.figure(1, figsize=(100, 60))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=96, azim=268)
    plt.cla()
    y = kmeans.labels_

    fig = px.scatter_3d(
        x_pca, x=0, y=1, z=2, color=y,
        title=f'Total Explained Variance: ',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()

    #  ONLY VISUAL TEXT, NOT IMPORTANT !
    # for name, label in [('First principal component ', 0), ('Second principal component', 1), ('Third principal component', 2)]:
    #     ax.text3D(x_pca[0].mean(),
    #               x_pca[1].mean(),
    #               x_pca[2].mean(),
    #               name,
    #               horizontalalignment='center',
    #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [0, 1, 2, 3, 4]).astype(np.float)


    # ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, cmap=plt.cm.nipy_spectral,
    #            edgecolor='k')
    #
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])

    # PCA 2D
    # plt.figure(figsize=(100, 60))
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], cmap='plasma')
    # plt.xlabel('First principal component')
    # plt.ylabel('Second Principal Component')

    plt.show()