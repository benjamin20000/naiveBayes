import pandas as pd
class CleanData:
    @staticmethod
    def clean(df, index_name):
        if isinstance(df.index, pd.RangeIndex):
            df = df.drop(columns=index_name)
        return df
