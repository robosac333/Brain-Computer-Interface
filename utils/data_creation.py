# %% Initialize the Spark session
import findspark
findspark.init()

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from datetime import timedelta
from pyspark.sql.functions import col
import os

from pyspark.sql.functions import lit

spark = SparkSession.builder \
    .appName("EEG_Analysis") \
    .master("local[*]") \
    .getOrCreate()

'''
Add location data to be used for the analysis
'''
base_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2"
body_parts = [(0,"Right_hand"), (1,"Left_hand")]

# %% Data Columns and Schema
'''
Dataframe selection and Schema
'''
#  Select relevant columns and drop NA values
columns_to_analyze = [
    "TimeStamp", "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10"]

from pyspark.sql.types import *

data_schema = [
                StructField('TimeStamp', TimestampType(), True),
                StructField('Delta_TP9', DoubleType(), True),
                StructField('Delta_AF7', DoubleType(), True),
                StructField('Delta_AF8', DoubleType(), True),
                StructField('Delta_TP10', DoubleType(), True),
                StructField('Theta_TP9', DoubleType(), True),
                StructField('Theta_AF7', DoubleType(), True),
                StructField('Theta_AF8', DoubleType(), True),
                StructField('Theta_TP10', DoubleType(), True),
                StructField('Alpha_TP9', DoubleType(), True),
                StructField('Alpha_AF7', DoubleType(), True),
                StructField('Alpha_AF8', DoubleType(), True),
                StructField('Alpha_TP10', DoubleType(), True),
                StructField('Beta_TP9', DoubleType(), True),
                StructField('Beta_AF7', DoubleType(), True),
                StructField('Beta_AF8', DoubleType(), True),
                StructField('Beta_TP10', DoubleType(), True),
                StructField('Gamma_TP9', DoubleType(), True),
                StructField('Gamma_AF7', DoubleType(), True),
                StructField('Gamma_AF8', DoubleType(), True),
                StructField('Gamma_TP10', DoubleType(), True)
                # StructField('label', IntegerType(), True)
            ]
final_struct = StructType(fields=data_schema)

# %% Load data from multiple CSV files
from pyspark.sql.functions import lit
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
        df = spark.read.csv(file_path, header=True, schema=final_struct).dropna()
        # df.shape = (df.count(), len(df.columns))
        # print(f"Loaded {file} of {part} with shape {df.shape}")
        dfs.append(df)

    # Union all DataFrames
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.union(df)
    combined_df = combined_df.withColumn("label", lit(index))
    return combined_df

# Load data for each body part
data_dict = {}
for index, part in body_parts:
    folder_path = os.path.join(base_path, part)
    data_dict[part] = load_data(index, folder_path, part)

# Show the shape of each DataFrame
for part, df in data_dict.items():
    print(f"{part} shape: {df.count(), len(df.columns)}")

# %% Combine the data from all body parts

# Initialize a variable to hold the combined DataFrame
combined_data = None

# Loop through the data_dict to union all the DataFrames
for part, df in data_dict.items():
    if combined_data is None:
        # Start with the first DataFrame
        combined_data = df
    else:
        # Union the next DataFrame
        combined_data = combined_data.union(df)

# Show the combined DataFrame
combined_data.shape = (combined_data.count(), len(combined_data.columns))

# print(f"Combined DataFrame shape: {combined_data.shape}")

# combined_data.show()

combined_data = combined_data.na.drop()
# Assuming 'df' is your DataFrame and 'timestamp_col' is the name of your timestamp column
# Convert the timestamp to ss.SSS format
dataset = combined_data.withColumn("TimeStamp", F.date_format("TimeStamp", "ss.SSS").cast(DoubleType()))

start_time = dataset.select('TimeStamp').collect()[0][0]
end_time = dataset.select('TimeStamp').collect()[-1][0]

# Step 3: Calculate the difference in milliseconds from the minimum timestamp
dataset = dataset.withColumn("TimeStamp",  F.round(F.col("TimeStamp") - F.lit(start_time), 3))

# Show the resulting DataFrame
# dataset.show(truncate=False)

# %% Average the concerned columns

from pyspark.sql.window import Window
# from pyspark.sql.functions import col, monotonically_increasing_id

granularity = 100 # Change this to your desired granularity in milliseconds

# Create a window
window_spec = Window.orderBy('TimeStamp').rowsBetween(0, granularity)

# Bucket the data by the time interval (granularity)
dataset = dataset.withColumn('TimeBucket', (F.col('TimeStamp') * 1000).cast('int'))

# Group by the buckets and calculate the average for each bucket
aggregations = {col: 'avg' for col in columns_to_analyze if col != 'TimeStamp'}
aggregations['TimeStamp'] = 'first'  # Keep the first TimeStamp for each bucket

# # Group by the buckets and calculate the average for each bucket
# aggregations = {col: 'avg' for col in columns_to_analyze[1:]}
agg_dataset = dataset.groupBy('TimeBucket').agg(aggregations)

# Rename the columns to remove "avg()" and "first()"
for col in columns_to_analyze:
    if col != 'TimeStamp':
        agg_dataset = agg_dataset.withColumnRenamed(f'avg({col})', col)
    else:
        agg_dataset = agg_dataset.withColumnRenamed(f'first({col})', col)

# Drop the TimeBucket column and reassign the DataFrame
agg_dataset = agg_dataset.drop('TimeBucket')

# Sort the aggregated dataset by TimeStamp to maintain order
agg_dataset = agg_dataset.orderBy('TimeStamp')

# Show the resulting dataset
# agg_dataset.show()

# Getting the shape of the dataset
# row_count = agg_dataset.count()
# column_count = len(agg_dataset.columns)

# print(f"Dataset shape: ({row_count}, {column_count})")

# %% Save the file to csv format

# Define the output path
output_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2\combined_data.csv"

pandas_df = agg_dataset.toPandas()

# Save the DataFrame to a CSV file
pandas_df.to_csv(output_path, index=False)

