import os
import pathlib
import sys
import numpy as np
import pandas as pd


from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns



def main ():

    data = pd.read_csv('out.csv',sep=',', header=0, index_col = None) ## always rename if cleaning    
    # print(data)
    # drop unneeded columns 
    scatter_data = data.dropna(subset=['t_max', 'precp', 'ECOZ_NAME'])
    counts = scatter_data.groupby(['year', 'month', 't_max', 'precp']).size().reset_index(name='fire_count') ## to get the counts
    
    # print(counts)
    
    ## Average max temp and total precipitation graph for amount of fires 
    ## if uncommented there are 33 graphs 

    # for year in sorted(counts['year'].unique()):
    #     data_year = counts[counts['year'] == year]
        
    #     plt.figure(figsize=(6, 5)) 
    #     sns.scatterplot(
    #         x='t_max',
    #         y='precp',
    #         data=data_year,
    #         hue='month',
    #         size='fire_count',
    #         palette='viridis',
    #         sizes=(20, 250),
    #         legend=True
    #     )

    #     plt.xlabel('Average Maximum Temperature')
    #     plt.ylabel('Total Precipitation')
    #     plt.title(f'Wildfires from {year} by Temperature, Precipitation & Month')
    #     plt.tight_layout()
    #     plt.show()
        
    
    ### STATS OF HOW TMAX AND PRECP AFFECT FIRE SIZE 
    # QUESTION: Does the average tmax per month and sum of precp indicate fire size
    
    print("QUESTION: How does the average maximum temperature per month and the monthly total of precipitation indicate fire size?")
    
    print("Correlation analysis of Weather Features and Hectares Burned")
    
    print("Scatter Plots of Weather Features and Hectares Burned")

    sns.scatterplot(x=scatter_data['t_max'], y=scatter_data['SIZE_HA'], palette='viridis')
    plt.xlabel('Average Maximum Temperature')
    plt.ylabel('Hectares Burned')
    plt.title(f'Maximum Temperature vs Hectares Burned')
    plt.tight_layout()
    plt.show()
    
    sns.scatterplot(x=scatter_data['precp'], y=scatter_data['SIZE_HA'],palette='viridis')
    plt.xlabel('Total Precipitation')
    plt.ylabel('Hectares Burned')
    plt.title(f'Precipation vs Hectares Burned')
    plt.tight_layout()
    plt.show()
    plt.show()
    
    print("Correlation analysis of Weather Features and Hectares Burned")
    
    pearson_corr = scatter_data[['t_max', 'precp', 'SIZE_HA']].corr(method='pearson')
    print("Pearson correlation:\n", pearson_corr)
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm')
    plt.title(f'Pearson Correlation')
    plt.show()

    # Spearman correlation
    spearman_corr = scatter_data[['t_max', 'precp', 'SIZE_HA']].corr(method='spearman')
    print("\nSpearman correlation:\n", spearman_corr)
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
    plt.title(f'Spearman Correlation')
    plt.show()
        

    ### STATS OF HOW TMAX AND PRECP AFFECT THE AMOUNT OF FIRES 
    ## chi squared
    # https://pandas.pydata.org/docs/user_guide/categorical.html - to split into cateogries so we can see what the chi squared says 
    
    print("QUESTION: How does the average maximum temperature per month and the monthly total of precipitation affect the amount of fires?")
    
    print("Chi Squared Analysis:")
    
    labels = ['Low', 'Medium', 'High']
    
    ## to put our data into bins 
    counts['tmax_bins'] = pd.cut(counts['t_max'], bins=3, labels=labels)
    counts['precp_bins'] = pd.cut(counts['precp'], bins=3, labels=labels)

    contingency = pd.crosstab(counts['tmax_bins'], counts['precp_bins'])

    chi = stats.chi2_contingency(contingency)
    print(f"p = {chi.pvalue}")
    
    ### FOR THE LAT AND LONG AND THE ECOZ_NAME PREDICT THE NUMBER OF FIRES PER MONTH IN THAT AREA
    
    print("QUESTION: Based on the location, can we predict the number of fires per month in that area?")
    
    print("Regression Analysis:")
    
    location_counts = scatter_data.groupby(['year', 'month', 't_max', 'precp', 'ECOZ_NAME', 'lat', 'long']).size().reset_index(name='fire_count')
    location_counts = location_counts.sort_values(by='ECOZ_NAME')
    
    features = ['year', 't_max', 'precp', 'lat', 'long', 'month']
    X = location_counts[features]
    y = location_counts['fire_count']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(450, max_depth=30, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("Random Forest Regressor Score on Number of Fires:")
    
    print(model.score(X_valid, y_valid))
    
    
    
    ### FOR THE LAT AND LONG AND THE ECOZ_NAME PREDICT THE HECTARE BURNED PER YEAR
    
    print("QUESTION: Based on the location, can we predict the hectares burned within that area?")
    
    print("Regression Analysis:")
    
    
    location_counts = scatter_data.groupby(['year', 'month', 't_max', 'precp', 'ECOZ_NAME', 'lat', 'long', 'SIZE_HA']).size().reset_index(name='fire_count')
    location_counts = location_counts.sort_values(by='ECOZ_NAME')
    
    features = ['year', 't_max', 'precp', 'lat', 'long', 'month', 'fire_count']
    X = location_counts[features]
    y = location_counts['SIZE_HA']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(450, max_depth=25, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("Random Forest Regressor Score on Fire Size:")
    
    print(model.score(X_valid, y_valid))
    
    model = KNeighborsRegressor(10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("K Nearest Neighbour Regressor Score on Fire Size:")
    
    print(model.score(X_valid, y_valid))
    
    
    ### FOR THE LAT AND LONG AND THE ECOZ_NAME PREDICT THE HECTARE CATEGORY
    
    print("Classification Analysis:")
    
    location_counts = scatter_data.groupby(['year', 'month', 't_max', 'precp', 'ECOZ_NAME', 'lat', 'long', 'SIZE_HA']).size().reset_index(name='fire_count')
    location_counts = location_counts.sort_values(by='ECOZ_NAME')
    
    labels = ['Small', 'Medium', 'Large']
    
    # https://stackoverflow.com/questions/67434627/how-can-i-split-my-data-in-pandas-into-specified-buckets-e-g-40-40-20
    location_counts['hectare_size'] = pd.qcut(location_counts['SIZE_HA'], q=3, labels=labels)
    
    features = ['year', 't_max', 'precp', 'lat', 'long', 'month']
    X = location_counts[features]
    y = location_counts['hectare_size']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(450, max_depth=25, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("Random Forest Classification Score and Classification Report on Fire Size Category:")
    
    print(model.score(X_valid, y_valid))
    print(classification_report(y_valid, y_pred))
    
    model = KNeighborsClassifier(10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("K Nearest Neighbour Classification Score and Classification Report on Fire Size Category:")
    
    print(model.score(X_valid, y_valid))
    print(classification_report(y_valid, y_pred))
    
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    print("Navie Bayes Score and Classification Report on Fire Size Category:")
    
    print(model.score(X_valid, y_valid))
    print(classification_report(y_valid, y_pred))
    
    return 


if __name__ == '__main__':
    main()
