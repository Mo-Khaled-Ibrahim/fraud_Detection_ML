# Databricks notebook source
# MAGIC %md
# MAGIC # 03: Model Training
# MAGIC
# MAGIC This notebook trains a fraud detection model using the processed features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install catboost shap
# MAGIC
# MAGIC import os
# MAGIC import sys
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import mlflow
# MAGIC import mlflow.catboost
# MAGIC
# MAGIC # Add src to path
# MAGIC sys.path.append('/Workspace/Users/mohamed.c.elshenity@gmail.com/fraud/src/model')
# MAGIC
# MAGIC from pyspark.sql import SparkSession
# MAGIC import pyspark.sql.functions as F
# MAGIC from train import train_model
# MAGIC from evaluate import evaluate_model
# MAGIC from threshold import find_best_threshold
# MAGIC from shap_explainer import compute_shap_reasons

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FraudDetectionModelTraining") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

print(f"Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Processed Data

# COMMAND ----------

# Load processed data
df = spark.table("fraud_features")

print(f"Processed data loaded: {df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split (Time-Based)

# COMMAND ----------

# Sort by timestamp
df = df.orderBy("unix_time")

# Calculate split points (70% train, 15% val, 15% test)
total_count = df.count()
train_end = int(total_count * 0.7)
val_end = int(total_count * 0.85)

# Add row index for splitting
from pyspark.sql.window import Window
window = Window.orderBy("unix_time")
df = df.withColumn("row_idx", F.row_number().over(window))

# Split
train_df = df.filter(F.col("row_idx") <= train_end)
val_df = df.filter((F.col("row_idx") > train_end) & (F.col("row_idx") <= val_end))
test_df = df.filter(F.col("row_idx") > val_end)

print(f"Train: {train_df.count():,} records")
print(f"Val: {val_df.count():,} records")
print(f"Test: {test_df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features and Labels

# COMMAND ----------

# Define feature columns
feature_cols = [
    "amt", "city_pop", "hour", "day_of_week", "month", "distance", "haversine_distance",
    "txn_count_1h", "txn_count_24h", "avg_amt_24h", "spend_24h", "unique_merchants_24h",
    "time_since_last_txn", "category", "merchant", "state", "gender", "city", "zip", "job"
]

# Select features that exist in the DataFrame
feature_cols = [col for col in feature_cols if col in df.columns]
label_col = "is_fraud"

print(f"Using {len(feature_cols)} features")

# COMMAND ----------

# Convert to pandas for model training (PANDAS BOUNDARY)
X_train = train_df.select(feature_cols).toPandas()
y_train = train_df.select(label_col).toPandas()[label_col]

X_val = val_df.select(feature_cols).toPandas()
y_val = val_df.select(label_col).toPandas()[label_col]

X_test = test_df.select(feature_cols).toPandas()
y_test = test_df.select(label_col).toPandas()[label_col]

print("Data converted to pandas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model with MLflow

# COMMAND ----------

# Train model
model = train_model(X_train, y_train, X_val, y_val, None)

# Find optimal threshold
best_threshold = find_best_threshold(model, X_val, y_val, n_folds=3)

# Evaluate on test set
metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)

print(f"Model trained!")
print(f"Test AUC: {metrics['auc']:.4f}")
print(f"Test F1: {metrics['f1']:.4f}")
print(f"Optimal threshold: {best_threshold:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop Spark Session

# COMMAND ----------

spark.stop()