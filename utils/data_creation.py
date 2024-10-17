import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md
from scipy import stats
from pyspark.sql import functions as F

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .appName("EEG_Analysis") \
    .master("local[*]") \
    .getOrCreate()

"""
Load data from a folder containing multiple CSV files.
"""
def load_data(index, folder_path, part):

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Initialize an empty list to store individual DataFrames
    dfs = []
    # Read each CSV file and create a DataFrame
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        df.shape = (df.count(), len(df.columns))
        print(f"Loaded {file} of {part} with shape {df.shape}")
        dfs.append(df)

    # Union all DataFrames
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.union(df)
    combined_df = combined_df.withColumn("label", lit(index))
    return combined_df

def create_dataset(start_time, end_time, cols, granularity=1000):
    timestamps = pd.date_range(start_time, end_time, freq=str(granularity)+'ms')
    data_table = spark.createDataFrame(index=timestamps, columns=cols)
    for col in cols[1:]:
        dataframe = dataframe.withColumn(col, F.lit(None))
    return data_table

def num_sampling(dataset, data_table, value_cols, aggregation='avg', granularity=1000):
    for i in range(data_table.count()):
        relevant_rows = dataset[
            (dataset['TimeStamp'] >= data_table.select("TimeStamp").head(1)[0][0]) &
            (dataset['TimeStamp'] < (data_table.select("TimeStamp").head(1)[0][0] +
                                    timedelta(milliseconds=1000)))] 
        for col in value_cols:
            # numerical cols which for the EEG data are the brain waves
            # We take the average value
            if len(relevant_rows) > 0:
                data_table.loc[data_table.index[i], str(col)] = np.average(relevant_rows[col])
            else:
                data_table.loc[data_table.index[i], str(col)] = np.nan 
    return data_table