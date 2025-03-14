from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Démarrage du script train.py...")

# Initialiser Spark
spark = SparkSession.builder.appName("MovieLensALS").getOrCreate()

# Chemins des fichiers
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/input/ml-100k/"
MODEL_PATH = "/home/imad/Desktop/movie_recommendation_system/model/als_model"

# Charger les données
try:
    logging.info("Chargement des données...")
    ratings = spark.read.csv(DATA_PATH + "u.data", sep="\t", inferSchema=True).toDF("user_id", "movie_id", "rating", "timestamp")
    logging.info("Données chargées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données : {e}")
    exit(1)

# Diviser les données en ensembles d'entraînement et de test
(training, test) = ratings.randomSplit([0.8, 0.2])

# Entraîner le modèle ALS
try:
    logging.info("Entraînement du modèle ALS...")
    als = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)
    logging.info("Modèle entraîné avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de l'entraînement du modèle : {e}")
    exit(1)

# Évaluer le modèle
try:
    logging.info("Évaluation du modèle...")
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    logging.info(f"Root Mean Squared Error (RMSE) : {rmse}")
except Exception as e:
    logging.error(f"Erreur lors de l'évaluation du modèle : {e}")
    exit(1)

# Sauvegarder le modèle
try:
    logging.info("Sauvegarde du modèle...")
    model.write().overwrite().save(MODEL_PATH)
    logging.info("Modèle sauvegardé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de la sauvegarde du modèle : {e}")
    exit(1)

# Arrêter Spark
spark.stop()
logging.info("Script train.py terminé.")