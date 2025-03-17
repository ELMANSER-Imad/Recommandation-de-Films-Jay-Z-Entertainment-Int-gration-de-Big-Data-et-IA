from flask import Flask, jsonify, request
from elasticsearch import Elasticsearch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting Flask API...")

# Initialize Flask app
app = Flask(__name__)

# Connexion à Elasticsearch
es = Elasticsearch(["http://elasticsearch:9200"])  # Utilisez "http://localhost:9200" localement
INDEX_NAME = "movie_recommendations"

# Route pour la page d'accueil
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue dans le système de recommandation de films !"}), 200

# Route pour obtenir des recommandations basées sur un film
@app.route("/recommend", methods=["GET"])
def recommend_movies():
    """Route pour obtenir des recommandations basées sur un film."""
    try:
        # Récupérer le titre du film depuis les paramètres de la requête
        movie_title = request.args.get('title')
        
        if not movie_title:
            return jsonify({"error": "Le paramètre 'title' est requis."}), 400
        
        # Rechercher le film dans Elasticsearch pour obtenir son ID
        query = {
            "query": {
                "match": {
                    "title": movie_title
                }
            }
        }
        response = es.search(index="movies_cleaned", body=query)
        
        if not response['hits']['hits']:
            return jsonify({"error": f"Aucun film trouvé avec le titre '{movie_title}'."}), 404
        
        movie_id = response['hits']['hits'][0]['_source']['movie_id']
        
        # Rechercher les utilisateurs ayant interagi avec ce film
        query = {
            "query": {
                "nested": {
                    "path": "recommendations",
                    "query": {
                        "match": {
                            "recommendations.movie_id": movie_id
                        }
                    }
                }
            }
        }
        response = es.search(index=INDEX_NAME, body=query)
        
        if not response['hits']['hits']:
            return jsonify({"error": f"Aucun utilisateur n'a interagi avec le film '{movie_title}'."}), 404
        
        # Récupérer les recommandations pour ces utilisateurs
        recommendations = []
        for hit in response['hits']['hits']:
            user_id = hit['_source']['user_id']
            user_recommendations = hit['_source']['recommendations']
            recommendations.append({
                "user_id": user_id,
                "recommendations": user_recommendations
            })
        
        return jsonify({"movie_id": movie_id, "recommendations": recommendations}), 200
    
    except Exception as e:
        logging.error(f"Une erreur s'est produite : {e}")
        return jsonify({"error": "Une erreur interne s'est produite."}), 500

# Route pour tester la connexion à Elasticsearch
@app.route("/test-elasticsearch", methods=["GET"])
def test_elasticsearch():
    if es.ping():
        return jsonify({"message": "Connexion à Elasticsearch réussie !"}), 200
    else:
        return jsonify({"error": "Impossible de se connecter à Elasticsearch."}), 500

# Start Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)