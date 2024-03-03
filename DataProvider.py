import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional


class DataProvider:
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_normalized: Optional[pd.DataFrame] = None
        self._load_and_clean_data()

    def _load_and_clean_data(self) -> None:
        # Load the data
        self.df_raw = pd.read_csv(self.file_path)
        self.df_raw.drop(['unix', 'date', 'symbol'], axis=1, inplace=True)

        # Reverse the DataFrame and reset index
        self.df_raw = self.df_raw.iloc[::-1].reset_index(drop=True)

        # Log transformation
        self.df_raw['volume'] = np.log(self.df_raw['volume'] + 1)
        self.df_raw['volume_from'] = np.log(self.df_raw['volume_from'] + 1)
        self.df_raw['tradecount'] = np.log(self.df_raw['tradecount'] + 1)

        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.df_raw)
        self.df_normalized = pd.DataFrame(normalized_data, columns=self.df_raw.columns)

    def get_raw_data(self) -> pd.DataFrame:
        if self.df_raw is not None:
            return self.df_raw
        else:
            raise ValueError("Raw data not loaded.")

    def get_normalized_data(self) -> pd.DataFrame:
        if self.df_normalized is not None:
            return self.df_normalized
        else:
            raise ValueError("Normalized data not available.")
