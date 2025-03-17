import unittest
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Spark session and load the model."""
        cls.spark = SparkSession.builder.appName("TestModel").getOrCreate()
        # Use absolute path to load the model
        cls.model = ALSModel.load("/home/imad/Desktop/movie_recommendation_system/model/als_model")

    @classmethod
    def tearDownClass(cls):
        """Stop Spark session."""
        cls.spark.stop()

    def test_recommendations(self):
        """Test if the model generates recommendations."""
        recommendations = self.model.recommendForUserSubset(self.spark.createDataFrame([(1,)], ["user_id"]), 10)
        self.assertGreater(len(recommendations.collect()), 0)

if __name__ == "__main__":
    unittest.main()