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
import seaborn as sns




def main ():
    
    ## plots and stats:
    ## scatter plot for max average and fire count and total percep and fire count 
    ## heat map 
    ## fire counts per year 
    
    data = pd.read_csv('out.csv',sep=',', header=0, index_col = None) ## always rename if cleaning
    
    scatter_data = data.dropna(subset=['t_max', 'precp'])

    grouped = scatter_data.groupby(['year', 'month', 't_max', 'precp']).size().reset_index(name='fire_count')

    plt.figure()
    

    plt.xlabel('Average Maximum Temperature (t_max)')
    plt.ylabel('Total Precipitation (precp)')
    plt.title('Number of Fires by Temperature, Precipitation, Month and Year')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    

    
    
    
    ## mannwhitney u test 
    
    ## check if t test is possible?
    
    ## chi test 
    
    
    ## make a map with the lat and long - shows where the fires are over years - different colours per year 
 
    ## change total wildfires monthly to have numbers for month instead of name and province into short form
    
    ## join on year month province or location? - canada wildfire and weather 
    
    
    ## how to analysis????
    
    return 


if __name__ == '__main__':
    main()

