import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from typing import List
from Trader import Trader  # Replace with your actual module name
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l1_l2
from keras.optimizers import Adam

class TradingModelTrainer:
    def __init__(self, trainData: pd.DataFrame, tradeData: pd.DataFrame, initial_capital: float, trade_size_dollars: float):
        self.data = tradeData
        self.trainData = trainData
        self.initial_capital = initial_capital
        self.trade_size_dollars = trade_size_dollars
        self.model_learning_history = []


    def create_lstm_model(self, timesteps, features):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, features)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(64, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def simulate_trading(self, model: Sequential, timesteps: int) -> Trader:
        """Simulates trading for a given model and returns the Trader instance."""
        prepared_data = self.prepare_data_for_prediction(self.trainData, timesteps)

        predictions = model.predict(prepared_data, batch_size=256, verbose="0")

        actions = np.argmax(predictions, axis=1)
        trader = Trader(self.data, self.trade_size_dollars, self.initial_capital)

        trader.trade(actions)

        return trader
    
    def get_learning_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_learning_history)

    def update_learning_history(self, final_capital: float, sharpe_ratio: float):
        self.model_learning_history.append({'final_capital': final_capital, 'sharpe_ratio': sharpe_ratio})

    def prepare_data_for_prediction(self, df: pd.DataFrame, timesteps: int):
            values = df.values
            X = []
            for i in range(len(values) - timesteps):
                X.append(values[i:(i + timesteps), :])

            return np.array(X)




