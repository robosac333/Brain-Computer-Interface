import os

from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import lit

from pyspark.sql import functions as F

from pyspark.sql.window import Window
"""
Load data from a folder containing multiple CSV files.
"""
def load_data(index, folder_path, part, spark, final_struct):

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

# %% Combine the data from all body parts

def combined_dataset(data_dict):
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

    return dataset

# %% Average the concerned columns

def average_out_timesteps(dataset, columns_to_analyze, granularity):
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

    return agg_dataset