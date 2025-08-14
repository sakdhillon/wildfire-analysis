# Wildfire Prediction and Analysis
Canada is known for the breathtaking scenery, from the beautiful landscapes, to the lakes and forests that this country has from coast to coast. Yet every year across Canada, on average 2.5 million hectares of our great forests burn. As the years progress, the hotter it has been in Canada, and more wildfires have occurred. This begs the question, what relation does our weather have when it comes to wildfires, and it is possible to use this to predict how bad or where a fire may occur?

This project explores the use of data science to combine wildfire and weather analysis in the hopes to provide early information to fire teams throughout the country in an effort to help improve response times and wildfire management.
The following questions were investigated:

    How does the average maximum temperature per month and the monthly total of precipitation indicate fire size?
    How does the average maximum temperature per month and the monthly total of precipitation affect the amount of fires?
    Based on the location, can we predict the number of fires per month in that area?
    Based on the location, can we predict the hectares burned within that area?

## Results 

Identified a significant relationship between temperature, precipitation and fire count with a resulting pvalue of 3.25 x $10^{-16}$. 
Trained a regression model explaining 32% of fire occurrences only using two weather features.
Trained a classification model correctly classifiying ~40% of fires based on 3 size categories.

## Requirements
Install the required libraries using:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy
  - pyspark

## Data to run
All the necessary data to run the wildfire_dataProcessing.py is held in the 'Data' file. The output of that processing file is used to run the wildfire_analysis.py file (with out.csv). To run extract_weather.py, this must be done on the SFU compute cluster which holds the GHCN Dataset used for this assignment. 

## How to run
To extract weather data (once on the compute cluster run):

    spark-submit extract_weather.py

To process data in the 'Data' file:

    python3 wildfire_dataProcessing.py ./data
    
To run analysis with the ouput file from data processing:

    python3 wildfire_analysis.py out.csv
