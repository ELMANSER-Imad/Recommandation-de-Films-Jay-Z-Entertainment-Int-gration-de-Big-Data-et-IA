from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from elasticsearch import Elasticsearch, helpers
import json
import logging
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Démarrage du script setup.py...")

# Initialiser la session Spark
spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()

# Chemins des fichiers
MODEL_PATH = "/home/imad/Desktop/movie_recommendation_system/model/als_model"
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/output/"
MOVIES_FILE = os.path.join(DATA_PATH, "movies_cleaned.csv")  # Répertoire
RATINGS_FILE = os.path.join(DATA_PATH, "ratings_cleaned.csv")  # Répertoire

# Vérifier l'existence des répertoires
if not os.path.exists(MOVIES_FILE) or not os.path.exists(RATINGS_FILE):
    logging.error("Les répertoires de données sont introuvables.")
    logging.error(f"Chemin des films : {MOVIES_FILE}")
    logging.error(f"Chemin des évaluations : {RATINGS_FILE}")
    spark.stop()
    exit(1)

# Charger le modèle entraîné
try:
    logging.info("Chargement du modèle ALS...")
    model = ALSModel.load(MODEL_PATH)
    logging.info("Modèle ALS chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    logging.error(f"Assurez-vous que le modèle existe dans : {MODEL_PATH}")
    spark.stop()
    exit(1)

# Charger les données utilisateurs et films
try:
    logging.info("Chargement des données utilisateurs et films...")
    movies = spark.read.csv(MOVIES_FILE, header=True, inferSchema=True)
    ratings = spark.read.csv(RATINGS_FILE, header=True, inferSchema=True)
    logging.info("Données chargées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données : {e}")
    logging.error(f"Assurez-vous que les fichiers CSV existent dans : {DATA_PATH}")
    spark.stop()
    exit(1)

# Générer des recommandations pour tous les utilisateurs
try:
    logging.info("Génération des recommandations...")
    user_recommendations = model.recommendForAllUsers(10)
    logging.info("Recommandations générées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de la génération des recommandations : {e}")
    spark.stop()
    exit(1)

# Initialiser Elasticsearch
try:
    es = Elasticsearch(["http://localhost:9200"], timeout=30)
    if not es.ping():
        raise Exception("Elasticsearch n'est pas accessible.")
    logging.info("Connexion à Elasticsearch réussie.")
except Exception as e:
    logging.error(f"Erreur lors de la connexion à Elasticsearch : {e}")
    spark.stop()
    exit(1)

INDEX_NAME = "movie_recommendations"

# Supprimer l'index s'il existe déjà
try:
    if es.indices.exists(index=INDEX_NAME):
        logging.info(f"Suppression de l'index existant '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)
except Exception as e:
    logging.error(f"Erreur lors de la suppression de l'index : {e}")
    spark.stop()
    exit(1)

# Créer un nouvel index
try:
    logging.info(f"Création de l'index '{INDEX_NAME}'...")
    es.indices.create(index=INDEX_NAME, body={
        "mappings": {
            "properties": {
                "user_id": {"type": "integer"},
                "recommendations": {"type": "nested", "properties": {
                    "movie_id": {"type": "integer"},
                    "rating": {"type": "float"}
                }}
            }
        }
    })
    logging.info(f"Index '{INDEX_NAME}' créé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de la création de l'index : {e}")
    spark.stop()
    exit(1)

# Insérer les recommandations dans Elasticsearch avec le Bulk API
try:
    logging.info("Indexation des recommandations dans Elasticsearch...")
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": row["user_id"],
            "_source": {
                "user_id": row["user_id"],
                "recommendations": [
                    {"movie_id": rec["movie_id"], "rating": rec["rating"]}
                    for rec in row["recommendations"]
                ]
            }
        }
        for row in user_recommendations.collect()
    ]
    helpers.bulk(es, actions)
    logging.info("Recommandations indexées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors de l'indexation des recommandations : {e}")
    spark.stop()
    exit(1)

# Arrêter la session Spark
spark.stop()
logging.info("Script setup.py terminé.")