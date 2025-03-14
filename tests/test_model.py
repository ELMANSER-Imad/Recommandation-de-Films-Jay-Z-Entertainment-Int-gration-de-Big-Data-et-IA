import unittest
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

class TestALSModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialiser la session Spark et charger le modèle."""
        cls.spark = SparkSession.builder.appName("TestALSModel").getOrCreate()
        cls.model = ALSModel.load("../model/als_model")

    @classmethod
    def tearDownClass(cls):
        """Arrêter la session Spark après les tests."""
        cls.spark.stop()

    def test_model_loading(self):
        """Teste si le modèle est correctement chargé."""
        self.assertIsNotNone(self.model, "Le modèle n'a pas été chargé correctement.")

    def test_model_predictions(self):
        """Teste si le modèle génère des prédictions valides."""
        # Créer un DataFrame de test
        test_data = self.spark.createDataFrame([(1, 101), (2, 102)], ["user_id", "movie_id"])
        predictions = self.model.transform(test_data)
        
        # Vérifier que les prédictions contiennent une colonne 'prediction'
        self.assertIn("prediction", predictions.columns, "La colonne 'prediction' est manquante.")

    def test_recommendations(self):
        """Teste si le modèle génère des recommandations valides."""
        recommendations = self.model.recommendForAllUsers(5)
        
        # Vérifier que les recommandations contiennent des données
        self.assertGreater(recommendations.count(), 0, "Aucune recommandation générée.")

# Exécuter les tests
if __name__ == '__main__':
    unittest.main()