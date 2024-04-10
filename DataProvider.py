import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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

        # Calculate technical indicators
        self.df_raw['rsi'] = self.calculate_rsi(self.df_raw['close'])
        self.df_raw['ema200'] = self.df_raw['close'].ewm(span=200, adjust=False).mean()
        self.df_raw['ema50'] = self.df_raw['close'].ewm(span=50, adjust=False).mean()
        self.df_raw['ema12'] = self.df_raw['close'].ewm(span=12, adjust=False).mean()
        self.df_raw['macd'] = self.calculate_macd(self.df_raw['close'])
        self.df_raw['atr'] = self.calculate_atr(self.df_raw)

        # Log transformation
        self.df_raw['volume'] = np.log(self.df_raw['volume'] + 1)
        self.df_raw['volume_from'] = np.log(self.df_raw['volume_from'] + 1)
        self.df_raw['tradecount'] = np.log(self.df_raw['tradecount'] + 1)

        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.df_raw)

        # Apply Min-Max scaling
        min_max_scaler = MinMaxScaler()
        min_max_normalized_data = min_max_scaler.fit_transform(normalized_data)
        self.df_normalized = pd.DataFrame(min_max_normalized_data, columns=self.df_raw.columns)

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series, fast_period=12, slow_period=26, signal=9):
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

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
        
    def plot_histograms(self) -> None:
        if self.df_raw is not None:
            num_columns = len(self.df_raw.columns)
            fig, axes = plt.subplots(num_columns, 1, figsize=(10, num_columns * 4))

            for i, col in enumerate(self.df_raw.columns):
                sns.histplot(self.df_raw[col], ax=axes[i], kde=True)
                axes[i].set_title(f'Histogram of {col}')

            plt.tight_layout()
            plt.show()

    def plot_histograms_norm(self) -> None:
        if self.df_normalized is not None:
            num_columns = len(self.df_normalized.columns)
            fig, axes = plt.subplots(num_columns, 1, figsize=(10, num_columns * 4))

            for i, col in enumerate(self.df_normalized.columns):
                sns.histplot(self.df_normalized[col], ax=axes[i], kde=True)
                axes[i].set_title(f'Histogram norm of {col}')

            plt.tight_layout()
            plt.show()