#https://www.machinelearningplus.com/pyspark/pyspark-outlier-detection-and-treatment/
# %% Import required libraries
import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, exp
from pyspark.sql.types import StructType
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("OutlierDetection").master("local[*]").getOrCreate()

from utils.outlier_detection import detect_continuous_variables, iqr_outlier_treatment, plot_outliers
from utils.data_creation import load_data, combined_dataset, average_out_timesteps
from utils.data_struct_init import final_struct, columns_to_analyze, data_schema

# %% Load the data
'''
Add location data to be used for the analysis
'''
base_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2"
body_parts = [(0,"Right_hand"), (1,"Left_hand")]

# Load data for each body part
data_dict = {}
for index, part in body_parts:
    folder_path = os.path.join(base_path, part)
    data_dict[part] = load_data(index, folder_path, part, spark, data_schema)

# Show the shape of each DataFrame
for part, df in data_dict.items():
    print(f"{part} shape: {df.count(), len(df.columns)}")

right_hand_data = data_dict['Right_hand']
left_hand_data = data_dict['Left_hand']

# %% Show the aggregated DataFrame
granularity = 6 # Chose this granularity as per TimeStamp column increments

agg_right_hand_dataset = average_out_timesteps(right_hand_data, columns_to_analyze, granularity)
agg_right_hand_dataset = agg_right_hand_dataset.withColumn("label", F.col("label").cast("int"))
agg_left_hand_dataset = average_out_timesteps(left_hand_data, columns_to_analyze, granularity)
agg_left_hand_dataset = agg_left_hand_dataset.withColumn("label", F.col("label").cast("int"))
agg_left_hand_dataset.show()

# %% Identifying continuous columns

continuous_columns = detect_continuous_variables(agg_right_hand_dataset, distinct_threshold=10, drop_vars=["TimeStamp", "label"])
print("Continuous columns:", continuous_columns)

# %% Check the shape of the aggregated DataFrame
print(agg_right_hand_dataset.count(), len(agg_right_hand_dataset.columns))
print(agg_left_hand_dataset.count(), len(agg_left_hand_dataset.columns))

# %% Apply IQR outlier treatment

righthand_outlier_treatment = iqr_outlier_treatment(agg_right_hand_dataset, continuous_columns, factor=1.5)
lefthand_outlier_treatment = iqr_outlier_treatment(agg_left_hand_dataset, continuous_columns, factor=1.5)

print("Original DataFrame row count:", agg_right_hand_dataset.count())
print("Outlier treated DataFrame row count:", righthand_outlier_treatment.count())
print("Original DataFrame row count:", agg_left_hand_dataset.count())
print("Outlier treated DataFrame row count:", lefthand_outlier_treatment.count())

# %% Check the shape of the outlier treated DataFrame
data_dict = {}
data_dict["Right_hand"] = righthand_outlier_treatment
data_dict["Left_hand"] = lefthand_outlier_treatment
combineddataset = combined_dataset(data_dict)
print(combineddataset.count(), len(combineddataset.columns))
# %% Combined DataFrame cache

combineddataset.cache()
# %% Visualise thr outliers vs non-outliers for Right hand data
import matplotlib.pyplot as plt

# Convert PySpark DataFrames to Pandas 
pdDF_righthand_outlier_treatment = righthand_outlier_treatment.toPandas()
pandas_agg_right_hand_dataset = agg_right_hand_dataset.toPandas()

pdDF_lefthand_outlier_treatment = lefthand_outlier_treatment.toPandas()
pandas_agg_left_hand_dataset = agg_left_hand_dataset.toPandas()

# %% Visualise thr outliers vs non-outliers for Right hand data
plot_outliers(pdDF_righthand_outlier_treatment, pandas_agg_right_hand_dataset, continuous_columns)

# %% Visualise thr outliers vs non-outliers for Left hand data
plot_outliers(pdDF_lefthand_outlier_treatment, pandas_agg_left_hand_dataset, continuous_columns)

# %% Saving the combined dataset to a csv file
pd_combined_dataset = combineddataset.toPandas()
pd_combined_dataset.to_csv(r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2\Outlier_filtered_dataset.csv", index=False)
# %%
