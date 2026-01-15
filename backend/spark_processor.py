"""
UIDAI CIDR BIG DATA PIPELINE - PERFORMANCE AGGREGATOR
Logic: PySpark / Apache Spark 3.x
Purpose: National-scale pre-aggregation of ECMP transaction logs.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def initialize_spark():
    return SparkSession.builder \
        .appName("UIDAI_Operational_Intelligence") \
        .config("spark.sql.broadcastTimeout", "3600") \
        .getOrCreate()

def calculate_psaci_spark(df):
    """
    Computes Pincode Service Access Concentration Index at Scale
    """
    # 1. Temporal Aggregation
    pin_stats = df.groupby("pincode").agg(
        F.sum("total_activity").alias("v_total"),
        F.sum("age_0_5").alias("c_total"),
        F.sum("age_5_17").alias("y_total")
    )
    
    # 2. Normalization via Windows
    v_max = pin_stats.select(F.max("v_total")).collect()[0][0]
    v_min = pin_stats.select(F.min("v_total")).collect()[0][0]
    
    # 3. Composite Calculation
    return pin_stats.withColumn(
        "psaci_score", 
        ((F.col("v_total") - v_min) / (v_max - v_min)) * 0.5 + 
        (((F.col("c_total") + F.col("y_total")) / F.col("v_total")) * 0.5)
    )

def main():
    spark = initialize_spark()
    
    # Loading from S3/HDFS Data Lake
    raw_df = spark.read.parquet("s3://uidai-cidr-logs/raw/year=2025/*")
    
    # Operational Filter
    processed_df = raw_df.filter(F.col("status") == "SUCCESS")
    
    # KPI Cubes
    district_cube = processed_df.groupby("postal_state", "postal_district") \
        .agg(F.sum("total_activity").alias("vol"), F.avg("total_activity").alias("avg_vol"))
    
    # Writing to High-Performance SQL Warehouse (Postgres/BigQuery)
    district_cube.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://cidr-warehouse-internal:5432/operational_db") \
        .mode("overwrite") \
        .save()

if __name__ == "__main__":
    main()
