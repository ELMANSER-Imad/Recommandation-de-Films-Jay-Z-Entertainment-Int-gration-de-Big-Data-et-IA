from flask import Flask, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Démarrage de l'API Flask...")

# Initialiser Flask
app = Flask(__name__)

# Initialiser Spark
spark = SparkSession.builder.appName("MovieRecommendationAPI").getOrCreate()

# Chemins des fichiers
MODEL_PATH = "/home/imad/Desktop/movie_recommendation_system/model/als_model"
DATA_PATH = "/home/imad/Desktop/movie_recommendation_system/data/output/"

# Charger le modèle ALS
try:
    logging.info("Chargement du modèle ALS...")
    model = ALSModel.load(MODEL_PATH)
    logging.info("Modèle ALS chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    exit(1)

# Charger les données nettoyées
try:
    logging.info("Chargement des données nettoyées...")
    movies = spark.read.parquet(DATA_PATH + "movies_cleaned.parquet")
    logging.info("Données chargées avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données : {e}")
    exit(1)

# Endpoint pour obtenir des recommandations
@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        recommendations = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["user_id"]), 10)
        recommended_movies = recommendations.collect()[0]["recommendations"]
        movie_titles = movies.select("movie_id", "title").rdd.collectAsMap()
        result = [
            {"movie_id": rec["movie_id"], "title": movie_titles.get(rec["movie_id"], "Unknown"), "predicted_rating": rec["rating"]}
            for rec in recommended_movies
        ]
        return jsonify({"user_id": user_id, "recommendations": result})
    except Exception as e:
        logging.error(f"Erreur lors de la génération des recommandations : {e}")
        return jsonify({"error": "Une erreur est survenue."}), 500

# Démarrer l'API
if __name__ == "__main__":
    app.run(port=5001, debug=False)