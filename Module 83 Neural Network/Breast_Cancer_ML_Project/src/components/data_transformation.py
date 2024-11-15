import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger

logger = get_logger()

class DataTransformation:
    def __init__(self, data_path: str):
        """
        Initialize with the path to the dataset.
        :param data_path: Path to the CSV file containing the dataset.
        """
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load the dataset from the provided path.
        :return: DataFrame containing the dataset.
        """
        try:
            logger.info(f"Loading dataset from {self.data_path}...")
            data = pd.read_csv(self.data_path)
            logger.info("Dataset loaded successfully.")
            return data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def split_data(self, data: pd.DataFrame):
        """
        Split the data into training and testing sets.
        :param data: DataFrame containing the dataset.
        :return: X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Splitting data into train and test sets...")
            X = data.drop(columns=["target"])
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info("Data split successfully.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Scale the features using StandardScaler.
        :param X_train: Training features.
        :param X_test: Testing features.
        :return: Scaled X_train and X_test.
        """
        try:
            logger.info("Scaling data using StandardScaler...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("Data scaling completed.")
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            raise

if __name__ == "__main__":
    data_path = "data/breast_cancer_data.csv"  # Path to your dataset

    # Initialize the transformation object
    transformer = DataTransformation(data_path)

    # Load the dataset
    data = transformer.load_data()

    # Split the dataset
    X_train, X_test, y_train, y_test = transformer.split_data(data)

    # Scale the dataset
    X_train_scaled, X_test_scaled = transformer.scale_data(X_train, X_test)

    logger.info("Data transformation pipeline executed successfully.")
