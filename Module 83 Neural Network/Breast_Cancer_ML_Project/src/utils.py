import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.logger import get_logger

logger = get_logger()

def save_dataset_to_csv(output_path="data"):
    """
    Load the breast cancer dataset from sklearn, process it, 
    and save it as a CSV file for later use.
    """
    try:
        # Load dataset
        logger.info("Loading breast cancer dataset from sklearn...")
        data = load_breast_cancer()

        # Convert to pandas DataFrame
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        df['target'] = data.target

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, "breast_cancer_data.csv")

        # Save as CSV
        logger.info(f"Saving dataset to {csv_path}...")
        df.to_csv(csv_path, index=False)
        logger.info("Dataset successfully saved as CSV.")

        return csv_path
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise

if __name__ == "__main__":
    save_dataset_to_csv()
