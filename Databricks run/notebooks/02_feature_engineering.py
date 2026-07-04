# Databricks notebook source
# MAGIC %md
# MAGIC # 02: Feature Engineering
# MAGIC
# MAGIC This notebook performs feature engineering on the raw transaction data and saves the processed data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Cell 3
import sys

# Add features directory to path
sys.path.append('/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/src/features')

from static_features import build_static_features
from window_features import build_window_features
from geospatial_features import build_geospatial_features

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FraudDetectionFeatureEngineering") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

print(f"Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Raw Data

# COMMAND ----------

# Define data paths
raw_data_path = "/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/parquet"

# Load raw data
df = spark.read.parquet(raw_data_path)

print(f"Raw data loaded: {df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Features

# COMMAND ----------

# Build static features
df = build_static_features(df)
print("Static features built")

# COMMAND ----------

# Build window features
df = build_window_features(df)
print("Window features built")

# COMMAND ----------

# Build geospatial features
df = build_geospatial_features(df)
print("Geospatial features built")

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Processed Data

# COMMAND ----------

# Save processed data to Unity Catalog table
table_name = "fraud_features"
df.write.mode("overwrite" ved to table: {table_name}")

# COMMAND ----------

spark.table("fraud_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop Spark Session

# COMMAND ----------

spark.stop()