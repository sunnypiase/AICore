import pandas as pd
import numpy as np
from typing import Tuple

class TradePosition:
    def __init__(self, position_type: str = "", entry_price: float = 0, contracts: float = 0):
        self.position_type = position_type
        self.entry_price = entry_price
        self.contracts = contracts

    def is_open(self) -> bool:
        return self.position_type == "long" or  self.position_type == "short"

class Trader:
    def __init__(self, df: pd.DataFrame, trade_size_dollars: float, initial_capital: float) -> None:
        self.df = df
        self.trade_size_dollars = trade_size_dollars
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = TradePosition()
        self.current_index = 0

        self.contract_sizes = self.df['close'].apply(lambda x: trade_size_dollars / x)
        self.trade_history = []
        self.trade_history_columns = ['index', 'entry_price', 'exit_price', 'type', 'contracts', 'profit_loss']

    def step(self, action: int) -> Tuple[float, bool]:
        results = 0
        done = False

        if self.current_capital <= self.trade_size_dollars:
            done = True
            return -100000, done

        position_contracts = self.contract_sizes[self.current_index]

        if action == 0:
            results = self.__handle_long_trade(position_contracts)
        elif action == 1:
            results = self.__wait()
        elif action == 2: 
            results = self.__handle_short_trade(position_contracts)

        self.current_index += 1
        if self.current_index >= len(self.df):
            done = True

        self.current_capital += results
        return results, done
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Converts the trade history list to a DataFrame."""
        return pd.DataFrame(self.trade_history, columns=self.trade_history_columns)
    
    def __open_position(self, position_type: str, position_contracts: float):
        self.position = TradePosition(position_type, self.df.iloc[self.current_index]['close'], position_contracts)
        # Add the opening of a position to the trade history list
        self.trade_history.append([
            self.current_index,
            self.df.iloc[self.current_index]['close'],
            np.nan,
            position_type,
            position_contracts,
            np.nan
        ])

    def __close_position(self) -> float:
        if not self.position.is_open():
            return 0
        exit_price = self.df.iloc[self.current_index]['close']

        if self.position.position_type == "long":
            profit_loss = (exit_price - self.position.entry_price) * self.position.contracts
        elif self.position.position_type == "short":
            profit_loss = (self.position.entry_price - exit_price) * self.position.contracts

        self.position = TradePosition()
        self.trade_history[-1][2] = self.df.iloc[self.current_index]['close']  # Update exit price
        self.trade_history[-1][5] = profit_loss  # Update profit/loss
        return profit_loss

    def __handle_long_trade(self, position_contracts: float) -> float:
        if self.position.is_open() and self.position.position_type == "long":
            # Already in a long position; no new action taken
            return 0
        elif self.position.is_open() and self.position.position_type == "short":
            # In a short position, close it before opening a new long position
            results = self.__close_position()
            self.__open_position("long", position_contracts)
            return results
        else:
            # Not in a position, open a new long position
            self.__open_position("long", position_contracts)
        return 0

    def __handle_short_trade(self, position_contracts: float) -> float:
        if self.position.is_open() and self.position.position_type == "short":
            # Already in a short position; no new action taken
            return 0
        elif self.position.is_open() and self.position.position_type == "long":
            # In a long position, close it before opening a new short position
            results = self.__close_position()
            self.__open_position("short", position_contracts)
            return results
        else:
            # Not in a position, open a new short position
            self.__open_position("short", position_contracts)
        return 0

    def __wait(self) -> float:
        return 0