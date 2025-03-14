from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, when, array, array_remove, avg, concat_ws
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_preprocessing.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logging.info("Starting data preprocessing...")

# Initialize Spark session
spark = SparkSession.builder.appName("MovieLensPreprocessing").getOrCreate()

# Define file paths
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/input/ml-100k/"
OUTPUT_PATH = "/home/imad/Desktop/movie_recommendation_system/data/output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

try:
    # Define custom schema for u.item
    item_schema = StructType([
        StructField("movie_id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("video_release_date", StringType(), True),
        StructField("IMDb_URL", StringType(), True),
        StructField("unknown", IntegerType(), True),
        StructField("Action", IntegerType(), True),
        StructField("Adventure", IntegerType(), True),
        StructField("Animation", IntegerType(), True),
        StructField("Children's", IntegerType(), True),
        StructField("Comedy", IntegerType(), True),
        StructField("Crime", IntegerType(), True),
        StructField("Documentary", IntegerType(), True),
        StructField("Drama", IntegerType(), True),
        StructField("Fantasy", IntegerType(), True),
        StructField("Film-Noir", IntegerType(), True),
        StructField("Horror", IntegerType(), True),
        StructField("Musical", IntegerType(), True),
        StructField("Mystery", IntegerType(), True),
        StructField("Romance", IntegerType(), True),
        StructField("Sci-Fi", IntegerType(), True),
        StructField("Thriller", IntegerType(), True),
        StructField("War", IntegerType(), True),
        StructField("Western", IntegerType(), True)
    ])

    # Load movies data with custom schema
    logging.info("Loading movies data with custom schema...")
    movies = spark.read.csv(DATA_PATH + "u.item", sep="|", schema=item_schema, encoding="ISO-8859-1")

    # Print existing column names
    print("Existing Column Names:", movies.columns)

    # Verify column count
    if len(movies.columns) != 24:
        raise ValueError(f"Expected 24 columns, but found {len(movies.columns)} in u.item.")

    # Load ratings data
    logging.info("Loading ratings data...")
    ratings = spark.read.csv(DATA_PATH + "u.data", sep="\t", inferSchema=True) \
        .toDF("user_id", "movie_id", "rating", "timestamp")

    # Load users data
    logging.info("Loading users data...")
    users = spark.read.csv(DATA_PATH + "u.user", sep="|", inferSchema=True) \
        .toDF("user_id", "age", "gender", "occupation", "zip_code")

    # Create 'genres' column
    logging.info("Creating 'genres' column...")
    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies = movies.withColumn("genres", array(*[when(col(c) == 1, c).otherwise(None) for c in genre_columns]))
    movies = movies.withColumn("genres", array_remove(col("genres"), None))

    # Convertir la colonne 'genres' en une chaîne de caractères
    movies = movies.withColumn("genres", concat_ws(",", col("genres")))

    # Calculate average rating per movie
    logging.info("Calculating average ratings...")
    avg_ratings = ratings.groupBy("movie_id").agg(avg("rating").alias("avg_rating"))

    # Merge movie data with average ratings
    logging.info("Merging movie data with average ratings...")
    movies = movies.join(avg_ratings, "movie_id", "left")

    # Save results in CSV format
    logging.info("Saving cleaned data...")
    movies.write.csv(OUTPUT_PATH + "movies_cleaned.csv", header=True, mode="overwrite")
    users.write.csv(OUTPUT_PATH + "users_cleaned.csv", header=True, mode="overwrite")
    ratings.write.csv(OUTPUT_PATH + "ratings_cleaned.csv", header=True, mode="overwrite")

    logging.info("Preprocessing completed. Cleaned files are saved.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
    spark.stop()
    exit(1)

finally:
    spark.stop()