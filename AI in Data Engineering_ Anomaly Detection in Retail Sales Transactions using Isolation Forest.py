# Databricks notebook source
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from sklearn.ensemble import IsolationForest
from pyspark.sql.types import LongType
from datetime import datetime
import numpy as np
import pandas as pd

# COMMAND ----------

spark = SparkSession.builder.appName("AnomalyDetection") \
     .config("spark.sql.execution.arrow.pyspark.enabled","true") \
     .config("spark.sql.shuffle.partitions", 3).getOrCreate()            

# COMMAND ----------

data = [
    {'customerID': 12, 'transactionId': 1, 'productIdList': [12, 54, 65]},
    {'customerID': 12, 'transactionId': 2, 'productIdList': [54, 12, 84]},
    {'customerID': 12, 'transactionId': 3, 'productIdList': [5665, 87, 34]},
    {'customerID': 13, 'transactionId': 1, 'productIdList': []},
    {'customerID': 13, 'transactionId': 2, 'productIdList': [12,20]},
    {'customerID': 14, 'transactionId': 1, 'productIdList': [12, 54, 65, 11]},
    {'customerID': 14, 'transactionId': 2, 'productIdList': [12, 54, 89]},
    {'customerID': 14, 'transactionId': 3, 'productIdList': [235, 652, 789]},
    {'customerID': 15, 'transactionId': 1, 'productIdList': [12, 54, 84 ,325]},
    {'customerID': 15, 'transactionId': 2, 'productIdList': [12, 54]}
]

# COMMAND ----------

schema = T.StructType([
    T.StructField("customerID", T.IntegerType(), True),
    T.StructField("transactionId", T.IntegerType(), True),
    T.StructField("productIdList", T.ArrayType(T.IntegerType()), True)
])

# COMMAND ----------

df = spark.createDataFrame(data,schema= schema)

# COMMAND ----------

df = df.withColumn("productIdList",F.when(F.col("productIdList").isNull(),F.array())
                                    .otherwise(F.col("productIdList")))

# COMMAND ----------

df = df.repartition(3)

# COMMAND ----------

spark.sql(
    """
  CREATE TABLE IF NOT EXISTS RetailSales_Trans (
  ID BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
  customerID INT,
  transactionId INT,
  productIdList ARRAY<INT>,
  AnomalyKey LONG,
  createddatetime TIMESTAMP
) USING DELTA
    """
)

# COMMAND ----------

spark.sql(
    """
  CREATE TABLE IF NOT EXISTS anomalous_transactions (
  ID BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1),
  customerID INT,
  transactionId INT,
  productIdList ARRAY<INT>,
  Anomaly Long,
  AnomalyKey LONG,
  createddatetime TIMESTAMP
) USING DELTA
    """
)

# COMMAND ----------

unique_customers = df.select('customerID').distinct().toPandas()['customerID'].tolist()

# COMMAND ----------

model = IsolationForest(contamination=0.2 , random_state=42)

# COMMAND ----------

anomaly_prediction_df = pd.DataFrame(columns=['customerID', 'transactionId','Anomaly'])

# COMMAND ----------

for customer_id in unique_customers:

    # Filter transactions for the current customer
    filtered_df = df.filter(df['customerID'] == customer_id)

    # Check for empty DataFrame
    if not filtered_df.rdd.isEmpty():

        # Extract product lists as a list of NumPy arrays
        customer_product_lists = filtered_df.select('productIdList').rdd.flatMap(lambda x: x).collect()

        # Convert list of NumPy arrays to 2D array (with padding)
        max_length = max(len(x) for x in customer_product_lists)
        customer_product_lists = np.vstack([x + [0] * (max_length - len(x)) for x in customer_product_lists])

        try:  
           # Train the model on the customer's transactions
           model.fit(customer_product_lists)
   
           # Add the anomaly prediction to the original DataFrame
           customer_anomaly_prediction = model.predict(customer_product_lists)
   
           # Create prediction DataFrame (handling potential empty transactionId list)
           customer_anomaly_prediction = pd.DataFrame({
               'customerID': [customer_id] * len(customer_anomaly_prediction),
               'transactionId': filtered_df.select('transactionId').rdd.flatMap(lambda x: x).collect(),
               'Anomaly': customer_anomaly_prediction})
   
           anomaly_prediction_df = pd.concat([anomaly_prediction_df, customer_anomaly_prediction], ignore_index=True)
   
        except Exception as e:
         print(f"Error during model fitting for customer {customer_id}: {e}")  
  
    else:
        # Handle the case where there are no transactions for the customer
        print(f"No transactions found for customer ID: {customer_id}")

# COMMAND ----------

df = df.join(spark.createDataFrame(anomaly_prediction_df), on=['customerID', 'transactionId'], how='left')

# COMMAND ----------

Total_records = df.withColumn("AnomalyKey", F.when(F.col('Anomaly') == -1, F.monotonically_increasing_id()) \
                  .otherwise(None)) \
                  .withColumn("createddatetime", F.current_timestamp())

# COMMAND ----------

Total_records = Total_records.drop('Anomaly')

# COMMAND ----------

Total_records.write.format('delta').option('mergeSchema','true').mode('append').saveAsTable('RetailSales_Trans')

# COMMAND ----------

print('\nRetailSales_Trans Table Contents')
spark.sql('select * from RetailSales_Trans order by ID , transactionid').show(truncate=False)

# COMMAND ----------

anomalous_transactions_df = df.filter(F.col('Anomaly') == -1) \
    .withColumn("createddatetime", F.current_timestamp()) \
    .withColumn("AnomalyKey", F.monotonically_increasing_id()) 

# COMMAND ----------

anomalous_transactions_df.write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable("anomalous_transactions")

# COMMAND ----------

print('\nRetailSales_Trans Table Contents')
spark.sql('select * from RetailSales_Trans order by ID , transactionid').show(truncate=False)

print('\nanomalous_transactions Table Contents')
spark.sql('select * from anomalous_transactions order by ID , transactionid').show(truncate=False)

# COMMAND ----------


