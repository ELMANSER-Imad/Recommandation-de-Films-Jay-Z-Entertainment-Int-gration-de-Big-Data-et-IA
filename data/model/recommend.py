from flask import Flask, jsonify, request
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from elasticsearch import Elasticsearch
import logging
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("recommend.log"),  # Log dans un fichier
        logging.StreamHandler()               # Log dans la console
    ]
)
logging.info("Démarrage du système de recommandation...")

# Initialiser Flask
app = Flask(__name__)

# Chemins absolus des fichiers
MODEL_PATH = "/home/imad/Desktop/movie_recommendation_system/model/als_model"
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/output/"
MOVIES_FILE = os.path.join(DATA_PATH, "movies_cleaned.csv")
RATINGS_FILE = os.path.join(DATA_PATH, "ratings_cleaned.csv")

# Initialiser Spark
spark = SparkSession.builder.appName("MovieRecommendationAPI").getOrCreate()

# Charger le modèle ALS
try:
    logging.info("Chargement du modèle ALS...")
    model = ALSModel.load(MODEL_PATH)
    logging.info("Modèle ALS chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    spark.stop()
    exit(1)

# Charger les données nettoyées
try:
    logging.info("Chargement des données nettoyées...")
    movies = spark.read.csv(MOVIES_FILE, header=True, inferSchema=True)
    ratings = spark.read.csv(RATINGS_FILE, header=True, inferSchema=True)
    logging.info("Données chargées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données : {e}")
    spark.stop()
    exit(1)

# Initialiser Elasticsearch
es = Elasticsearch(["http://localhost:9200"])
INDEX_NAME = "movie_recommendations"

# Endpoint pour obtenir des recommandations pour un utilisateur spécifique
@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        # Vérifier si l'utilisateur existe dans les données
        user_exists = ratings.filter(col("user_id") == user_id).count() > 0
        if not user_exists:
            return jsonify({"error": f"Utilisateur avec ID {user_id} introuvable."}), 404

        # Générer des recommandations pour l'utilisateur
        recommendations = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["user_id"]), 10)
        recommended_movies = recommendations.collect()[0]["recommendations"]

        # Mapper les IDs de films aux titres
        movie_titles = movies.select("movie_id", "title").rdd.collectAsMap()
        result = [
            {"movie_id": rec["movie_id"], "title": movie_titles.get(rec["movie_id"], "Unknown"), "predicted_rating": rec["rating"]}
            for rec in recommended_movies
        ]

        return jsonify({"user_id": user_id, "recommendations": result})

    except Exception as e:
        logging.error(f"Erreur lors de la génération des recommandations : {e}")
        return jsonify({"error": "Une erreur est survenue."}), 500

# Endpoint pour interroger Elasticsearch
@app.route('/search', methods=['GET'])
def search_movies():
    try:
        query = request.args.get('query')
        if not query:
            return jsonify({"error": "Le paramètre 'query' est requis."}), 400

        # Rechercher des films dans Elasticsearch
        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "genres"]
                }
            }
        }
        response = es.search(index=INDEX_NAME, body=es_query)
        hits = response["hits"]["hits"]

        # Formater les résultats
        results = [
            {
                "movie_id": hit["_source"]["movie_id"],
                "title": hit["_source"]["title"],
                "genres": hit["_source"]["genres"],
                "avg_rating": hit["_source"].get("avg_rating", None)
            }
            for hit in hits
        ]

        return jsonify({"query": query, "results": results})

    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans Elasticsearch : {e}")
        return jsonify({"error": "Une erreur est survenue."}), 500

# Démarrer l'API
if __name__ == "__main__":
    try:
        logging.info("Démarrage de l'API Flask...")
        app.run(port=5001, debug=False)
    except Exception as e:
        logging.error(f"Erreur lors du démarrage de l'API : {e}")
    finally:
        # Arrêter Spark
        spark.stop()
        logging.info("Script recommend.py terminé.")