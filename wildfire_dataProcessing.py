import os
import pathlib
import sys
import numpy as np
import pandas as pd


## middle of province locations
def prov_location(data):
    # https://www.latlong.net/category/provinces-40-60.html
    
    lat_map = {
        'BC': 53.726669,
        'AB': 55.000000,
        'SK': 55.000000,
        'MB': 56.415211,
        'ON': 50.000000,
        'QC': 53.000000,
        'NS': 45.000000,
        'NB': 46.498390,
        'NL': 53.135509,
        'NWT': 64.8255,
        'YT': 64.2823,
        'PEI': 46.250000
    }
    
    long_map = {
        'BC': -127.647621,
        'AB': -115.000000,
        'SK': -106.000000,
        'MB': -98.739075,
        'ON': -85.000000,
        'QC': -70.000000,
        'NS': -63.000000,
        'NB': -66.159668,
        'NL': -57.660435,
        'NWT': -124.8457,
        'YT': -135.0000,
        'PEI': -63.000000
    }
    
    data['lat'] = data['prov'].map(lat_map)
    data['long'] = data['prov'].map(long_map)
    
    return data 


## checking the closest distance for station and fire to get the corret weather values 
# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def distance(fires, weather):
    
    ### check by year and month so we don't match random ones 
    stations = weather[(weather['year'] == fires['year']) & (weather['month'] == fires['month']) & (weather['tmax_avg'].notna()) & (weather['precp_sum'].notna())]
    
    lat1 = fires['lat']
    lon1 = fires['long']
    
    lat2 = stations['latitude']
    lon2 = stations['longitude']
    
    r = 6371 # km
    p = np.pi / 180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    s = 2 * r * np.arcsin(np.sqrt(a))
    
    return s

## returns the best value you can find for 'avg_tmax' and 'precp_sum' for one city from the list of all the weather stations (returns the closest distance?)
def best_weather (city, stations):
    distances = distance(city, stations)

    closest_station = distances.argmin()
    
    tmax = stations['tmax_avg'].iloc[closest_station]
    precp = stations['precp_sum'].iloc[closest_station]
    return tmax, precp

def main():
    # all from one directory
    input_directory = pathlib.Path(sys.argv[1])
    
    ## read all the files needing processing
    wildfires = pd.read_csv(input_directory / 'CANADA_WILDFIRES.csv', sep=',', header=0, index_col = None, parse_dates=['REP_DATE'])
    weather = pd.read_parquet(input_directory / 'weather-sub/weather.parquet') ## for some reason doesn't print all the numbers 
    fire_nums = pd.read_csv(input_directory / 'Total_Wildfires_Monthly.csv', sep='\t', skiprows=1, header=0, encoding='utf-16')  
    fire_causes = pd.read_csv(input_directory / 'Wildfire_Causes.csv', sep='\t', skiprows=1, header=0, encoding='utf-16')  

      
    ## WILDFIRES 
    ## clean the canada_wildfire csv - get month and year - combine based on that 
    
    wildfires = wildfires.rename(columns={'SRC_AGENCY': 'prov'})
    
    wildfires['year'] = wildfires['REP_DATE'].dt.year
    wildfires['month'] = wildfires['REP_DATE'].dt.month
    
    ## starting at 1990 - 2025 (smallest sequence all files have)
    wildfires = wildfires[(wildfires['year'] >= 1990) & (wildfires['year'] <= 2025)]
    wildfires['year'] = wildfires['year'].astype(int)
    wildfires['month'] = wildfires['month'].astype(int)
    
    ## drop unneeded columns - FID was just the row number 
    wildfires = wildfires.drop(['FID', 'REP_DATE', 'PROTZONE'], axis=1)
    
    wildfires = wildfires.astype({'LATITUDE': 'float', 'LONGITUDE': 'float', 'SIZE_HA': 'float'})
    
    wildfires['lat'] = np.floor(wildfires['LATITUDE'])
    wildfires['long'] = np.floor(wildfires['LONGITUDE'])
    
    wildfires = wildfires.drop(['LATITUDE', 'LONGITUDE'], axis=1)
    
    ## rename parks to be part of a province since most files have province locations
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
    
    # print(wildfires)
    
    ## FIRE NUMS
    ## change the years to be ints and not floats
    ## jurisdiction to be the short form 
    ## months to be numbered
    
    ## map provinces to be written in short form and months to be by number 
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

    # drop unneeded data and map necessary columns 
    fire_nums = fire_nums.drop(['Data Qualifier'], axis=1)
    fire_nums['prov'] = fire_nums['prov'].map(prov_map)
    fire_nums['month'] = fire_nums['month'].map(month_map)
    fire_nums = prov_location(fire_nums)
    fire_nums['month'] = fire_nums['month'].astype('Int64')
    
    fire_nums = fire_nums.fillna(0)
    
    # print(fire_nums)
    
    ## melt so we have a column for year 
    fire_nums = pd.melt(fire_nums, id_vars=['prov', 'month', 'lat', 'long'], var_name='year', value_name='fire_num')
    
    # print(fire_nums)
    
    
    
    ## FIRE CAUSES
    ## change the years to be ints and not floats
    ## jurisdiction to be the short form
    
    ## change the cause to be in short form 
    cause_map = {
        'Human activity' : 'H',
        'Lightning' : 'L',
        'Natural cause' : 'N',
        'Prescribed burn' : 'PB',
        'Reburn' : 'R',
        'Unspecified' : 'U'
    }
    
    fire_causes = fire_causes.rename(columns={'Jurisdiction': 'prov'})
    
    # drop unneeded data and map necessary columns 
    fire_causes = fire_causes.drop(['Data Qualifier'], axis=1)
    fire_causes['prov'] = fire_causes['prov'].map(prov_map)
    fire_causes['Cause'] = fire_causes['Cause'].map(cause_map)
    fire_causes = prov_location(fire_causes)
    fire_causes = fire_causes.fillna(0)
    
    ## melt so we have a column for year 
    fire_causes = pd.melt(fire_causes, id_vars=['prov', 'Cause', 'lat', 'long'], var_name='year', value_name='fire_num')

    # print(fire_causes)
    # print (weather)
    
    ## divide by 10 since that is what GHCN distributes
    weather = weather.dropna(subset=['tmax_avg', 'precp_sum'])
    weather['tmax_avg'] = weather['tmax_avg'] / 10
    
    
    ## https://stackoverflow.com/questions/33204763/how-to-pass-multiple-arguments-to-the-apply-function
    wildfires['stuff'] = wildfires.apply(lambda fire : best_weather(fire, weather), axis = 1)
    wildfires['t_max'] = wildfires['stuff'].apply(lambda pair: pair[0])
    wildfires['precp'] = wildfires['stuff'].apply(lambda pair: pair[1])
    
    wildfires = wildfires.drop(['stuff'], axis=1)
    
    # print(wildfires) 
    
    ## returns a file of joined data for wildfire_analysis.py
    wildfires.to_csv('out.csv', index=False)
    
    
    return 


if __name__ == '__main__':
    main()

