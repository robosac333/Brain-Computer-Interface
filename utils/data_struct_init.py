# Data Columns and Schema
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