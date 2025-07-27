import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

class load_data:
    """load data from local files or kaggle api."""

    @staticmethod
    def load_csv(model_name):
        """load csv file from local data folder using model name."""
        path = f"data/{model_name}.csv"
        if not os.path.exists(path):
            raise RuntimeError("path does not exist")
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            raise RuntimeError("file not opening") from e

    @staticmethod
    def load_phishing_data_by_api():
        """load phishing dataset from kaggle using kagglehub."""
        file_path = "phishing.csv"
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "eswarchandt/phishing-website-detector",
            file_path,
        )
        return df
