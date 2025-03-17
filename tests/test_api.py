import unittest
import requests

class TestAPI(unittest.TestCase):
    BASE_URL = "http://localhost:5000"

    def test_recommend_endpoint(self):
        """Test the /recommend endpoint."""
        # Test with a valid movie title
        response = requests.get(f"{self.BASE_URL}/recommend?title=Toy%20Story")
        self.assertEqual(response.status_code, 200)
        self.assertIn("movie_id", response.json())
        self.assertIn("recommendations", response.json())

        # Test with a missing title parameter
        response = requests.get(f"{self.BASE_URL}/recommend")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

        # Test with a non-existent movie title
        response = requests.get(f"{self.BASE_URL}/recommend?title=NonExistentMovie")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())

if __name__ == "__main__":
    unittest.main()