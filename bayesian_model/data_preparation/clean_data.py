import pandas as pd

class clean_data:
    """handle basic dataframe cleaning tasks."""

    @staticmethod
    def clean(df, index_name):
        """drop index column if index is a range index."""
        if isinstance(df.index, pd.RangeIndex):
            df = df.drop(columns=index_name)
        return df
