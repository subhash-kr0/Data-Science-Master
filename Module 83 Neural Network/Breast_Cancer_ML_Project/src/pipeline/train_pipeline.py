import os
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import get_logger

logger = get_logger()

def run_training_pipeline():
    try:
        logger.info("Starting training pipeline...")

        # Paths
        data_path = "data/breast_cancer_data.csv"

        # Step 1: Data Transformation
        logger.info("Initializing DataTransformation...")
        transformer = DataTransformation(data_path)

        # Load and preprocess data
        data = transformer.load_data()
        X_train, X_test, y_train, y_test = transformer.split_data(data)
        X_train_scaled, X_test_scaled = transformer.scale_data(X_train, X_test)

        # Step 2: Model Training
        logger.info("Initializing ModelTrainer...")
        trainer = ModelTrainer()

        # Train the model
        model = trainer.train_model(X_train_scaled, y_train)

        # Evaluate the model
        evaluation_metrics = trainer.evaluate_model(model, X_test_scaled, y_test)
        logger.info(f"Evaluation Metrics: {evaluation_metrics}")

        # Save the model
        trainer.save_model(model)
        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    run_training_pipeline()
