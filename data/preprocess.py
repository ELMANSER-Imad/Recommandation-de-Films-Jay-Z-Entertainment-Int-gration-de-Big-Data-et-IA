from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, when, array, array_remove, avg, concat_ws, to_date
import os
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logging.info("Démarrage du prétraitement des données...")

# Initialiser Spark
spark = SparkSession.builder.appName("MovieLensPreprocessing").getOrCreate()

# Chemins des fichiers
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/input/ml-100k/"
OUTPUT_PATH = "/home/imad/Desktop/movie_recommendation_system/data/output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

try:
    # Schéma personnalisé pour u.item
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

    # Charger les données avec le schéma spécifié
    movies = spark.read.csv(DATA_PATH + "u.item", sep="|", schema=item_schema, encoding="ISO-8859-1")
    ratings = spark.read.csv(DATA_PATH + "u.data", sep="\t", inferSchema=True).toDF("user_id", "movie_id", "rating", "timestamp")
    users = spark.read.csv(DATA_PATH + "u.user", sep="|", inferSchema=True).toDF("user_id", "age", "gender", "occupation", "zip_code")

    # Créer la colonne 'genres'
    genre_columns = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies = movies.withColumn("genres", array(*[when(col(c) == 1, c).otherwise(None) for c in genre_columns]))
    movies = movies.withColumn("genres", array_remove(col("genres"), None))
    movies = movies.withColumn("genres", concat_ws(",", col("genres")))

    # Convertir 'release_date' en format date
    movies = movies.withColumn("release_date", to_date(col("release_date"), "dd-MMM-yyyy"))

    # Calculer la note moyenne par film
    avg_ratings = ratings.groupBy("movie_id").agg(avg("rating").alias("avg_rating"))
    movies = movies.join(avg_ratings, "movie_id", "left")

    # Sauvegarder les données nettoyées
    movies.write.csv(OUTPUT_PATH + "movies_cleaned.csv", header=True, mode="overwrite")
    users.write.csv(OUTPUT_PATH + "users_cleaned.csv", header=True, mode="overwrite")
    ratings.write.csv(OUTPUT_PATH + "ratings_cleaned.csv", header=True, mode="overwrite")

    logging.info("Prétraitement terminé avec succès.")

except Exception as e:
    logging.error(f"Erreur lors du prétraitement : {e}")
    spark.stop()
    exit(1)

finally:
    spark.stop()