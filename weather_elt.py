import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.FloatType()),
    # year is only present because of the Hive partitioning
    types.StructField('year', types.IntegerType(), nullable=True),
    types.StructField('date', types.DateType()),  # becomes a types.DateType in the output ## YYYYMMDD 
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('name', types.StringType()),
])

def main(in_directory, out_directory):

    weather = spark.read.json(in_directory, schema=observation_schema)
    
    # extract year and month
    weather = weather.withColumn("month", functions.month("date"))
    weather = weather.drop('date')
    
    weather = weather.withColumn('lat_floor', functions.floor('latitude'))
    weather = weather.withColumn('long_floor', functions.floor('longitude'))
    
    weather = weather.drop('latitude', 'longitude','station')
    
    weather.show()
    
    # combine data via year and month and location - average temp and average percp 
    group = weather.groupBy('lat_floor', 'long_floor', 'year', 'month', 'observation')
    weather_grouped = group.agg(functions.avg(weather['value']).alias('average_val'))

    weather_grouped.show()
    num_rows = weather_grouped.count()
    
    print(num_rows)

    weather_grouped.write.json(out_directory, compression='gzip', mode='overwrite') ## write to one file
    
    

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
