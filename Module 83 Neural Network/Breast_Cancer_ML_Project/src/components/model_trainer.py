import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.logger import get_logger

logger = get_logger()

class ModelTrainer:
    def __init__(self, model_output_path="artifacts"):
        """
        Initialize the ModelTrainer class.
        :param model_output_path: Directory to save the trained model.
        """
        self.model_output_path = model_output_path
        os.makedirs(self.model_output_path, exist_ok=True)

    def train_model(self, X_train, y_train):
        """
        Train the RandomForestClassifier on the training data.
        :param X_train: Scaled training features.
        :param y_train: Training labels.
        :return: Trained model.
        """
        try:
            logger.info("Initializing RandomForestClassifier...")
            model = RandomForestClassifier(random_state=42)

            logger.info("Training the model...")
            model.fit(X_train, y_train)

            logger.info("Model training completed.")
            return model
        except Exception as e:
            logger.error(f"Error training the model: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model on the test data.
        :param model: Trained model.
        :param X_test: Scaled test features.
        :param y_test: Test labels.
        :return: Dictionary with evaluation metrics.
        """
        try:
            logger.info("Evaluating the model...")
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            logger.info(f"Model Accuracy: {accuracy:.2f}")
            logger.info(f"Classification Report:\n{report}")

            return {"accuracy": accuracy, "report": report}
        except Exception as e:
            logger.error(f"Error evaluating the model: {e}")
            raise

    def save_model(self, model, model_name="random_forest_model.pkl"):
        """
        Save the trained model to the specified directory.
        :param model: Trained model.
        :param model_name: Name of the model file.
        """
        try:
            model_path = os.path.join(self.model_output_path, model_name)
            logger.info(f"Saving model to {model_path}...")
            joblib.dump(model, model_path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise

if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    # Paths and setup
    data_path = "data/breast_cancer_data.csv"
    transformer = DataTransformation(data_path)

    # Load and preprocess data
    data = transformer.load_data()
    X_train, X_test, y_train, y_test = transformer.split_data(data)
    X_train_scaled, X_test_scaled = transformer.scale_data(X_train, X_test)

    # Train and evaluate the model
    trainer = ModelTrainer()
    model = trainer.train_model(X_train_scaled, y_train)
    evaluation_metrics = trainer.evaluate_model(model, X_test_scaled, y_test)

    # Save the model
    trainer.save_model(model)
