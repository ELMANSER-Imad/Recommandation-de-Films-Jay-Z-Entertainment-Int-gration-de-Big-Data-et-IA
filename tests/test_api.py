import unittest
import requests

# URL de base de l'API
BASE_URL = "http://localhost:5000"

class TestMovieRecommendationAPI(unittest.TestCase):
    def test_missing_title_parameter(self):
        """Teste si l'API retourne une erreur lorsque le paramètre 'title' est manquant."""
        response = requests.get(f"{BASE_URL}/recommend")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Le paramètre 'title' est requis.")

    def test_movie_not_found(self):
        """Teste si l'API retourne une erreur lorsque le film n'est pas trouvé."""
        response = requests.get(f"{BASE_URL}/recommend?title=NonExistentMovie")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Aucun film trouvé avec le titre 'NonExistentMovie'.")

    def test_valid_recommendations(self):
        """Teste si l'API retourne des recommandations valides pour un film existant."""
        # Remplacez "Toy Story" par un titre de film présent dans votre dataset
        response = requests.get(f"{BASE_URL}/recommend?title=Toy Story")
        self.assertEqual(response.status_code, 200)
        self.assertIn("movie_id", response.json())
        self.assertIn("recommendations", response.json())
        self.assertIsInstance(response.json()["recommendations"], list)

# Exécuter les tests
if __name__ == '__main__':
    unittest.main()