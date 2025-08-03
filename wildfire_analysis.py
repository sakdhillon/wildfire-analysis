import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import skimage
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



def main():
    
    filename1 = sys.argv[1]
    # filename2 = sys.argv[2]
    # filename3 = sys.argv[3]
    
    wildfires = pd.read_csv(filename1, sep=',', header=0, index_col = None, parse_dates=['REP_DATE'])
    # fire_nums = pd.read_csv(filename2, sep=' ', header=0, index_col=2)    
    
    ## clean the canada_wildfire csv - get month and year - combine based on that  - lat and long floor 
    # toDrop = wildfires['FID']
    wildfires = wildfires.rename(columns={'SRC_AGENCY': 'prov'})
    
    wildfires['year'] = wildfires['REP_DATE'].dt.year
    wildfires['month'] = wildfires['REP_DATE'].dt.month
    
    wildfires = wildfires.drop(['FID', 'REP_DATE', 'PROTZONE'], axis=1)
    
    print(wildfires)
    
    wildfires = wildfires.astype({'LATITUDE': 'float', 'LONGITUDE': 'float', 'SIZE_HA': 'float'})
    
    wildfires['lat'] = np.floor(wildfires['LATITUDE'])
    wildfires['long'] = np.floor(wildfires['LONGITUDE'])
    
    wildfires = wildfires.drop(['LATITUDE', 'LONGITUDE'], axis=1)
    
    print(wildfires)
 
    ## change total wildfires monthly to have numbers for month instead of name and province into short form
    
    ## join on year month province or location? - canada wildfire and weather 
    
    
    ## how to analysis????
    
    
    return 


if __name__ == '__main__':
    main()

