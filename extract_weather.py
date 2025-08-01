## code from https://github.sfu.ca/ggbaker/cluster-datasets/tree/main/weather as provided by Greg Baker 
## edits made by: Serena 

import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('GHCN extracter').getOrCreate()

ghcn_path = '/courses/datasets/ghcn-repartitioned'
ghcn_stations = '/courses/datasets/ghcn-more/ghcnd-stations.txt'
output = 'ghcn-subset'


observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),  # becomes a types.DateType in the output
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
    # year is only present because of the Hive partitioning
    types.StructField('year', types.IntegerType(), nullable=True),
])


station_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('name', types.StringType()),
])


def station_data(line):
    return [line[0:11].strip(), float(line[12:20]), float(line[21:30]), float(line[31:37]), line[41:71].strip()]


def main():
    ## Stations data...
    stations_rdd = spark.sparkContext.textFile(ghcn_stations).map(station_data)
    stations = spark.createDataFrame(stations_rdd, schema=station_schema).hint('broadcast')

    ## Observations data...
    obs = spark.read.csv(ghcn_path, header=None, schema=observation_schema)

    ## Filter as we like...
    # keep only some years
    obs = obs.filter((obs['year'] >= 1990) & (obs['year'] <= 2025))
    
    obs = obs.filter(functions.isnull(obs['qflag']))
    obs = obs.drop(obs['mflag']).drop(obs['qflag']).drop(obs['sflag']).drop(obs['obstime'])
    obs = obs.filter(obs['observation'].isin('TMAX', 'PRCP'))
    obs = obs.filter(obs.station.startswith('CA'))

    # parse the date string into a real date object
    obs = obs.withColumn('newdate', functions.to_date(obs['date'], 'yyyyMMdd'))
    obs = obs.drop('date').withColumnRenamed('newdate', 'date')   
    
    # optional, if you want the station locations joined in...
    obs = obs.join(stations, on='station')

    obs.write.json(output, mode='overwrite', compression='gzip')
    

main()
