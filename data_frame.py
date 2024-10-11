import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

from utils.data_creation import load_data

# Load the data path
base_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset"
body_parts = [(0,"Right_hand"), (1,"Left_hand"), (2,"Right_leg"), (3,"Left_leg")]


# Load data for each body part
data_dict = {}
for index, part in body_parts:
    folder_path = os.path.join(base_path, part)
    data_dict[part] = load_data(index, folder_path, part)

#  Select relevant columns and drop NA values
columns_to_analyze = [
    "TimeStamp","Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10", "label"
]

# Creating dataframes for each body part (Pandas)
right_hand_pd = data_dict["Right_hand"].select(columns_to_analyze).dropna().toPandas()
left_hand_pd = data_dict["Left_hand"].select(columns_to_analyze).dropna().toPandas()
right_leg_pd = data_dict["Right_leg"].select(columns_to_analyze).dropna().toPandas()
left_leg_pd = data_dict["Left_leg"].select(columns_to_analyze).dropna().toPandas()

# Creating dataframes for each body part (Spark)
right_hand_pd = data_dict["Right_hand"].select(columns_to_analyze).dropna()
left_hand_pd = data_dict["Left_hand"].select(columns_to_analyze).dropna()
right_leg_pd = data_dict["Right_leg"].select(columns_to_analyze).dropna()
left_leg_pd = data_dict["Left_leg"].select(columns_to_analyze).dropna()