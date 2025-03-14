from flask import Blueprint, request, jsonify
from elasticsearch import Elasticsearch
import json

# Créer un Blueprint pour les routes
routes_bp = Blueprint('routes', __name__)

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])
INDEX_NAME = "movie_recommendations"

@routes_bp.route('/recommend', methods=['GET'])
def recommend_movies():
    """Route pour obtenir des recommandations basées sur un film."""
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