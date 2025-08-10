import os
import pathlib
import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import skimage
import matplotlib.pyplot as plt


def main():
    
    input_directory = pathlib.Path(sys.argv[1])
    # output_directory = pathlib.Path(sys.argv[2])
    
    wildfires = pd.read_csv(input_directory / 'wildfire.csv', sep=',', header=0, index_col = None, parse_dates=['REP_DATE'])
    weather = pd.read_parquet(input_directory / 'weather-sub/weather.parquet') ## for some reason doesn't print all the numbers 
    fire_nums = pd.read_csv(input_directory / 'Total_Wildfires_Monthly.csv', sep='\t', skiprows=1, header=0, encoding='utf-16')  
    fire_causes = pd.read_csv(input_directory / 'Wildfire_Causes.csv', sep='\t', skiprows=1, header=0, encoding='utf-16')  
      
    ## WILDFIRES 
    ## clean the canada_wildfire csv - get month and year - combine based on that  - lat and long floor 
    
    wildfires = wildfires.rename(columns={'SRC_AGENCY': 'prov'})
    
    wildfires['year'] = wildfires['REP_DATE'].dt.year
    wildfires['month'] = wildfires['REP_DATE'].dt.month
    
    wildfires = wildfires.drop(['FID', 'REP_DATE', 'PROTZONE'], axis=1)
    
    print(wildfires)
    
    wildfires = wildfires.astype({'LATITUDE': 'float', 'LONGITUDE': 'float', 'SIZE_HA': 'float'})
    
    wildfires['lat'] = np.floor(wildfires['LATITUDE'])
    wildfires['long'] = np.floor(wildfires['LONGITUDE'])
    
    wildfires = wildfires.drop(['LATITUDE', 'LONGITUDE'], axis=1)
    
    parks_map = {
        'BC': 'BC',
        'AB': 'AB',
        'SK': 'SK',
        'MB': 'MB',
        'ON': 'ON',
        'QC': 'QC',
        'NS': 'NS',
        'NB': 'NB',
        'NL': 'NL',
        'NWT': 'NWT',
        'YT': 'YT',
        'PC-BA': 'AB',
        'PC-BP': 'ON',
        'PC-BT': 'SK',
        'PC-CB': 'NS',
        'PC-CH': 'SK',
        'PC-CT': 'BC',
        'PC-EI': 'AB',
        'PC-FO': 'QC',
        'PC-FR': 'BC',
        'PC-FU': 'NB',
        'PC-FW': 'SK',
        'PC-GB': 'ON',
        'PC-GF': 'BC',
        'PC-GH': 'BC',
        'PC-GI': 'QC',
        'PC-GL': 'BC',
        'PC-GM': 'NL',
        'PC-GR': 'SK',
        'PC-JA': 'AB',
        'PC-KE': 'NS',
        'PC-KG': 'NB',
        'PC-KL': 'YT',
        'PC-KO': 'BC',
        'PC-LL': 'AB',
        'PC-LM': 'QC',
        'PC-LO': 'NS',
        'PC-MI': 'QC',
        'PC-NA': 'NWT',
        'PC-PA': 'SK',
        'PC-PE': 'PEI',
        'PC-PP': 'ON',
        'PC-PR': 'BC',
        'PC-PU': 'ON',
        'PC-RE': 'BC',
        'PC-RM': 'MB',
        'PC-RO': 'AB',
        'PC-SL': 'ON',
        'PC-SY': 'NWT',
        'PC-TI': 'ON',
        'PC-TN': 'NL',
        'PC-VU': 'YT',
        'PC-WB': 'AB',
        'PC-WL': 'AB',
        'PC-WP': 'MB',
        'PC-YO': 'BC'
    }
    
    wildfires['prov'] = wildfires['prov'].map(parks_map)
    
    print(wildfires)
    
    ## FIRE NUMS
    ## change the years to be ints and not floats
    ## jurisdiction to be the short form 
    ## months to be numbered???
    
    prov_map = {
        'British Columbia': 'BC',
        'Alberta': 'AB',
        'Saskatchewan': 'SK',
        'Manitoba': 'MB',
        'Ontario': 'ON',
        'Quebec': 'QC',
        'Nova Scotia': 'NS',
        'New Brunswick': 'NB',
        'Newfoundland and Labrador': 'NL',
        'Northwest Territories': 'NWT',
        'Yukon': 'YT',
        'Prince Edward Island': 'PEI'
    }
    
    month_map = {
        'January' : 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12,
    }
    fire_nums = fire_nums.rename(columns={'Jurisdiction': 'prov'})
    fire_nums = fire_nums.rename(columns={'Month': 'month'})

    fire_nums = fire_nums.drop(['Data Qualifier'], axis=1)
    fire_nums['prov'] = fire_nums['prov'].map(prov_map)
    fire_nums['month'] = fire_nums['month'].map(month_map)
    fire_nums['month'] = fire_nums['month'].astype('Int64')
    
    fire_nums = fire_nums.fillna(0)
    
    print(fire_nums)
    
    
    ## FIRE CAUSES
    ## change the years to be ints and not floats
    ## jurisdiction to be the short form
    
    cause_map = {
        'Human activity' : 'H',
        'Lightning' : 'L',
        'Natural cause' : 'N',
        'Prescribed burn' : 'PB',
        'Reburn' : 'R',
        'Unspecified' : 'U'
    }
    
    fire_causes = fire_causes.rename(columns={'Jurisdiction': 'prov'})
    fire_causes = fire_causes.drop(['Data Qualifier'], axis=1)
    fire_causes['prov'] = fire_causes['prov'].map(prov_map)
    fire_causes['Cause'] = fire_causes['Cause'].map(cause_map)
    fire_causes = fire_causes.fillna(0)
    
    print(fire_causes)
    
    
    # weather = weather.drop(['Data Qualifier'], axis=1)
    print (weather)
    
    ## make a map with the lat and long - shows where the fires are over years - different colours per year 
 
    ## change total wildfires monthly to have numbers for month instead of name and province into short form
    
    ## join on year month province or location? - canada wildfire and weather 
    
    
    ## how to analysis????
    
    
    return 


if __name__ == '__main__':
    main()

