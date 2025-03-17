from elasticsearch import Elasticsearch
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Création des indices dans Elasticsearch...")

# Initialiser Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Configuration des indices
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
    },
    "mappings": {
        "properties": {
            "movie_id": {"type": "integer"},
            "title": {"type": "text"},
            "genres": {"type": "text"},
            "release_date": {"type": "date"},
            "avg_rating": {"type": "float"}
        }
    }
}

# Créer l'index pour les films
MOVIES_INDEX = "movies"
if not es.indices.exists(index=MOVIES_INDEX):
    es.indices.create(index=MOVIES_INDEX, body=INDEX_SETTINGS)
    logging.info(f"Index '{MOVIES_INDEX}' créé avec succès.")
else:
    logging.info(f"Index '{MOVIES_INDEX}' existe déjà.")

logging.info("Création des indices terminée.")