from pyspark.sql.types import IntegerType, StringType, NumericType
from pyspark.sql.functions import approxCountDistinct
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

def detect_continuous_variables(df, distinct_threshold, drop_vars = []):
    continuous_columns = []
    for column in df.drop(*drop_vars).columns:
        dtype = df.schema[column].dataType
        if isinstance(dtype, (IntegerType, NumericType)):
            distinct_count = df.select(approxCountDistinct(column)).collect()[0][0]
            if distinct_count > distinct_threshold:
                continuous_columns.append(column)
    return continuous_columns



from pyspark.sql.functions import col, exp
def iqr_outlier_treatment(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter outliers and update the DataFrame
        dataframe = dataframe.filter((F.col(column) >= lower_bound) & (F.col(column) <= upper_bound))

    return dataframe

def plot_outliers(pdDF_outlier_treatment, pdDF, continuous_columns):
    # Loop through each column in continuous_columns
    for col_name in continuous_columns:
        plt.figure(figsize=(12, 6))  # Create a new figure for each column
        
        # Create boxplot for the original data
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        pdDF.boxplot(column=col_name, grid=False)
        plt.title(f'Original - {col_name}')
        plt.xticks(rotation=90)
        
        # Create boxplot for the treated data
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        pdDF_outlier_treatment.boxplot(column=col_name, grid=False)
        plt.title(f'Treated - {col_name}')
        plt.xticks(rotation=90)
        
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()  # Display the plot
