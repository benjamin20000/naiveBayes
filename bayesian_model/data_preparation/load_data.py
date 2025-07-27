import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

class LoadData:
    # func for loading the local data
    @staticmethod
    def load_csv(model_name):
        path = f"data/{model_name}.csv"
        if not os.path.exists(path):
            raise RuntimeError("path does not exist")
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            raise RuntimeError("file not opening") from e


    @staticmethod
    ##func for loading the external data by api
    def load_phishing_data_by_api():
        file_path = "phishing.csv"
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "eswarchandt/phishing-website-detector",
            file_path,
        )
        return df

