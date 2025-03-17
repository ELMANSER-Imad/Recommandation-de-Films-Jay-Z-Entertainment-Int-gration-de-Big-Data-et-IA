from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Démarrage de l'entraînement du modèle ALS...")

# Initialiser Spark
spark = SparkSession.builder.appName("ALSModelTraining").getOrCreate()

# Chemins des fichiers
DATA_PATH = "../output"
MODEL_PATH = "als_model/"

try:
    # Charger les données
    ratings = spark.read.csv(DATA_PATH + "ratings_cleaned.csv", header=True, inferSchema=True)

    # Diviser les données en ensembles d'entraînement et de test
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Entraîner le modèle ALS
    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        rank=10,
        maxIter=10,
        regParam=0.1
    )
    model = als.fit(training)

    # Évaluer le modèle
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    logging.info(f"RMSE du modèle : {rmse}")

    # Sauvegarder le modèle
    model.write().overwrite().save(MODEL_PATH)
    logging.info("Modèle ALS sauvegardé avec succès.")

except Exception as e:
    logging.error(f"Erreur lors de l'entraînement ou de la sauvegarde du modèle : {e}")
    spark.stop()
    exit(1)

finally:
    spark.stop()