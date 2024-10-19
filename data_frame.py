# %% Initialize the Spark session
import findspark
findspark.init()

from pyspark.sql import SparkSession

import os

from utils.data_struct_init import final_struct, columns_to_analyze

from utils.data_creation import load_data, combined_dataset, average_out_timesteps

spark = SparkSession.builder \
    .appName("EEG_Analysis") \
    .master("local[*]") \
    .getOrCreate()

'''
Add location data to be used for the analysis
'''
base_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2"
body_parts = [(0,"Right_hand"), (1,"Left_hand")]

# Load data for each body part
data_dict = {}
for index, part in body_parts:
    folder_path = os.path.join(base_path, part)
    data_dict[part] = load_data(index, folder_path, part, spark, final_struct)

# Show the shape of each DataFrame
for part, df in data_dict.items():
    print(f"{part} shape: {df.count(), len(df.columns)}")

# %% Combine the data from all body parts

combined_dataset = combined_dataset(data_dict)

granularity = 100 # Chose this granularity as per TimeStamp column increments

aggregated_dataset = average_out_timesteps(combined_dataset, columns_to_analyze, granularity)

# %% Save the file to csv format

# Define the output path
output_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2\combined_data.csv"

pandas_df = aggregated_dataset.toPandas()

# Save the DataFrame to a CSV file
pandas_df.to_csv(output_path, index=False)
