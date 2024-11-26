# %% Initialize the Spark session
import findspark
findspark.init()

from pyspark.sql import SparkSession

import os

from utils.data_struct_init import final_struct, columns_to_analyze, data_schema

from utils.data_creation import load_data, combined_dataset, average_out_timesteps

spark = SparkSession.builder \
    .appName("EEG_Analysis") \
    .master("local[*]") \
    .getOrCreate()

'''
Add location data to be used for the analysis
'''
base_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2"
body_parts = [(0,"Left_hand"), (1,"Right_hand")]

# Load data for each body part
data_dict = {}
for index, part in body_parts:
    folder_path = os.path.join(base_path, part)
    data_dict[part] = load_data(index, folder_path, part, spark, data_schema)

# Show the shape of each DataFrame
for part, df in data_dict.items():
    print(f"{part} shape: {df.count(), len(df.columns)}")

## %% Combine the data from all body parts
combined_dataset = combined_dataset(data_dict)

combined_dataset.cache()
# test 
print(combined_dataset.count(), len(combined_dataset.columns))
# label_analysis = combined_dataset.select("label")
# labelcount = label_analysis.filter(label_analysis['label'] >= 0.5).count()
# print(labelcount)9
combined_dataset.show()

# %% Show the aggregated DataFrame
granularity = 5 # Chose this granularity as per TimeStamp column increments

aggregated_dataset = average_out_timesteps(combined_dataset, columns_to_analyze, granularity)

aggregated_dataset.show()
# %% Save the file to csv format
print(aggregated_dataset.count(), len(aggregated_dataset.columns))

# %% Save the file to csv format

# Define the output path
output_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2\combined_data.csv"

pandas_df = combined_dataset.toPandas()

# Save the DataFrame to a CSV file
pandas_df.to_csv(output_path, index=False)
