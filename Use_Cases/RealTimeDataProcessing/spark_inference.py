from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Create Spark Session
spark = SparkSession \
    .builder \
    .appName("KafkaSparkStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()


# Start with fields that don't follow the pattern
schema_fields = [
    StructField("Time", StringType(), True)
]
schema_fields += [StructField(f"Amount", StringType(), True)]
# Add fields V1 to V28
schema_fields += [StructField(f"V{i}", StringType(), True) for i in range(1, 29)]

# Create the schema
schema = StructType(schema_fields)

# Read data from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fraud-message") \
    .load()

# Process the Kafka data
df = df.selectExpr("CAST(value AS STRING) as json")
df = df.select(from_json(col("json"), schema).alias("data")).select("data.*")


for column in df.columns:
    #if column != "Time":  # Assuming 'Time' is not a feature to be casted
    df = df.withColumn(column, col(column).cast(FloatType()))

feature_columns = df.columns


#print("DataFrame schema before VectorAssembler:")
#df.printSchema()

model_path = "trained_model/"  
model = PipelineModel.load(model_path)

# Make predictions
predictions = model.transform(df)


# Since it's a streaming context, we use writeStream to output predictions
query = predictions.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \ 
    .option("numRows", 3) \    r
    .start()

query.awaitTermination()


# Perform transformations
#transformed_df = df.select("V28")

# Output the result to the console
#query = transformed_df \
#    .writeStream \
#    .outputMode("append") \
#    .format("console") \
#    .start()


query.awaitTermination()