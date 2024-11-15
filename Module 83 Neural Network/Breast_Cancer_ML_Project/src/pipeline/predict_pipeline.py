import os
import joblib
import numpy as np
from src.logger import get_logger

logger = get_logger()

class PredictionPipeline:
    def __init__(self, model_path="artifacts/random_forest_model.pkl"):
        """
        Initialize PredictionPipeline with the path to the trained model.
        :param model_path: Path to the saved model file.
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the trained model from the specified path.
        :return: Loaded model.
        """
        try:
            logger.info(f"Loading model from {self.model_path}...")
            model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data: np.ndarray):
        """
        Predict using the trained model.
        :param input_data: Numpy array of input features.
        :return: Predictions from the model.
        """
        try:
            logger.info("Making predictions...")
            predictions = self.model.predict(input_data)
            logger.info("Predictions completed.")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

if __name__ == "__main__":
    try:
        # Example input data (replace with real input)
        input_features = np.array([[15.78, 17.89, 102.34, 785.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 
                                    1.2, 1.4, 7.8, 120.3, 0.4, 0.3, 0.2, 0.4, 0.3, 0.2, 
                                    18.12, 22.45, 120.45, 800.3, 0.12, 0.24, 0.15, 0.32, 0.25, 0.14]])

        # Initialize PredictionPipeline
        pipeline = PredictionPipeline()

        # Make predictions
        predictions = pipeline.predict(input_features)
        logger.info(f"Predicted Class: {predictions}")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise
