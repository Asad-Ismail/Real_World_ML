from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id


# Create Spark Session
spark = SparkSession \
    .builder \
    .appName("FraudDetectionModelTrainingXGBoost") \
    .getOrCreate()

# Load feature data
features_path = "creditcardfraud/X_train.csv"  
df_features = spark.read.csv(features_path, header=True, inferSchema=True)

# Load label data
labels_path = "creditcardfraud/y_train.csv"  
df_labels = spark.read.csv(labels_path, header=True, inferSchema=True)


# Add a monotonically increasing id to both dataframes
df_features = df_features.withColumn("row_id", monotonically_increasing_id())
df_labels = df_labels.withColumn("row_id", monotonically_increasing_id())

# Join on the row id
df_joined = df_features.join(df_labels, "row_id").drop("row_id")


feature_columns = df_joined.columns
feature_columns.remove("Class")  
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")


# Classifier
classifier = RandomForestClassifier(labelCol="Class", featuresCol="features")

# Pipeline
pipeline = Pipeline(stages=[assembler, classifier])


print(f"Training model!!")
# Fit the model
model = pipeline.fit(df_joined)

# Save the model
model_path = "trained_model/"  # Update this path
model.write().overwrite().save(model_path)

#df_joined.show(1)
