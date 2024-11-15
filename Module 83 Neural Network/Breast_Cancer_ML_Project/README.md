# **Breast Cancer ML Project**

## **Project Overview**
This project aims to build a machine learning pipeline to predict breast cancer using the `load_breast_cancer` dataset from scikit-learn. The dataset contains features extracted from digitized images of breast masses, which are used to classify tumors as benign or malignant.

---

## **Directory Structure**
```plaintext
breast_cancer_ml_project/
├── data/                        # Stores the dataset in CSV format
├── notebooks/                   # Jupyter notebooks for experimentation
├── src/                         # Source code for the project
│   ├── __init__.py              # Initialization file
│   ├── logger.py                # Logging utility
│   ├── exception.py             # Custom exception class
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   ├── components/              # Core components of the pipeline
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Code for ingesting data
│   │   ├── data_transformation.py # Code for transforming data
│   │   ├── model_trainer.py     # Code for training the ML model
│   ├── pipeline/                # Pipeline scripts
│       ├── __init__.py
│       ├── train_pipeline.py    # Training pipeline
│       ├── predict_pipeline.py  # Prediction pipeline
├── import_data.py               # Script to import and save dataset
├── setup.py                     # Setup script for packaging
├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation
├── LICENSE                      # License for the project
├── .gitignore                   # Files and directories to ignore in Git
└── project.log                  # Log file
```

