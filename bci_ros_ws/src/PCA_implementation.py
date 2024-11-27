# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.PCA.html
# %% Import required libraries
import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, exp
from pyspark.sql.types import StructType
from pyspark.sql import functions as F

from utils.data_struct_init import final_struct, columns_to_analyze, data_schema_otherwise

# 2. Memory configuration
spark = SparkSession.builder.config("spark.sql.shuffle.partitions", "4").config("spark.driver.memory", "2g").getOrCreate()

file_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\Trial_2\Outlier_filtered_dataset.csv"

df = spark.read.csv(file_path, header=True, schema=StructType(data_schema_otherwise))

# Fix: Reduce partitions
df = df.coalesce(1)  # or small number like 2-4

# Show the shape of the DataFrame
print(f"Shape: {df.count(), len(df.columns)}")


# %% Now applying PCA on the data

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Convert the data to dense vector
feature_columns = columns_to_analyze[1:-1]

# Assemble the columns to a single column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Transform the data
df = assembler.transform(df)

# Joint the labels with the features
df = df.select("features" , "label")

# First split the dataframe
train_df, test_df = df.randomSplit([0.7, 0.3], seed=123)

# Method 1: Keep as separate dataframes
X_train = train_df.select("features")
Y_train = train_df.select("label")
X_test = test_df.select("features")
Y_test = test_df.select("label")

print("Size of the Training Datasets:", "Training", X_train.count(), Y_train.count())
print("Size of the Testing Datasets:", "Training", X_test.count(), Y_test.count())

# %% Apply PCA
pca = PCA(k=2, inputCol="features", outputCol="pca_features")

model = pca.fit(X_train)

X_train = model.transform(X_train).select("pca_features")

X_test = model.transform(X_test).select("pca_features")

X_train.cache()
X_test.cache()
Y_train.cache()
Y_test.cache()

# %% Now we can use the PCA features to train the Logistic Regression model
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# First, we need to combine features and labels back into single dataframes
# Create training dataframe with features and label
train_data = X_train.select("pca_features").join(Y_train.select("label"))
test_data = X_test.select("pca_features").join(Y_test.select("label"))

# Initialize Logistic Regression
lr = LogisticRegression(
    featuresCol="pca_features",  # PCA transformed features
    labelCol="label",
    maxIter=100,    # maximum iterations
    regParam=0.1    # regularization parameter
)

# Fit the model
lr_model = lr.fit(train_data)

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Additional evaluation metrics
print("Training summary:")
training_summary = lr_model.summary
print(f"Area under ROC: {training_summary.areaUnderROC}")
print(f"F-measure: {training_summary.fMeasureByLabel()}")

# Print model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(range(len(lr_model.coefficients)), lr_model.coefficients):
    print(f"Feature {feature}: {coef}")

# %% Save the model to disk

file_path = r"C:\Users\sachi\pyspark_tutorial\muse_pipeline\Telepathic-Navigation\muse_dataset\lr_model"

# Save the model
lr_model.save(file_path)

# Or with overwrite option
lr_model.write().overwrite().save(file_path)

# %% Load the model from disk when needed
loaded_model = LogisticRegressionModel.load(file_path)

# Use the loaded model
predictions = loaded_model.transform(test_data)

# %% Plot the predictions against the actual labels

import matplotlib.pyplot as plt
import numpy as np

# Convert the predictions and labels to numpy arrays
predictions = np.array(predictions.select("prediction").collect())
labels = np.array(predictions.select("label").collect())

# Plot the predictions against the actual labels

plt.figure(figsize=(10, 6)) 
plt.plot(predictions, label='Predictions')
plt.plot(labels, label='Actual Labels')
plt.title("Predictions vs Actual Labels")
plt.legend()
plt.show()

# %% Extending the code to classify data using Random Forest

from pyspark.ml.classification import RandomForestClassifier

# Initialize Random Forest

rf = RandomForestClassifier(
    featuresCol="pca_features",  # PCA transformed features
    labelCol="label",
    maxDepth=5,    # maximum depth of the tree
    numTrees=20    # number of trees in the forest
)

# Fit the model
rf_model = rf.fit(train_data)

# Make predictions on test data
predictions = rf_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# %% 