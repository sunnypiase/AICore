import pandas as pd
import numpy as np
from typing import List, Tuple

class TradePosition:
    def __init__(self, position_type: str = "", entry_price: float = 0, contracts: float = 0):
        self.position_type = position_type
        self.entry_price = entry_price
        self.contracts = contracts

    def is_open(self) -> bool:
        return self.position_type in ["long", "short"]

class Trader:
    def __init__(self, df: pd.DataFrame, trade_size_dollars: float, initial_capital: float) -> None:
        self.close_prices = df['close'].values  # Convert to NumPy array
        self.trade_size_dollars = trade_size_dollars
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = TradePosition()
        self.current_index = 0
        self.contract_sizes = trade_size_dollars / df['close']  # Use the NumPy array
        self.trade_history = []


    def trade(self, actions: List[int]) -> None:
        # Convert contract sizes to a numpy array for faster access
        contract_sizes_np = self.contract_sizes.values

        for i, action in enumerate(actions):
            self.current_index = i
            if self.current_capital <= self.trade_size_dollars:
                break

            position_contracts = contract_sizes_np[self.current_index]

            # Inline the handle_trade logic for long and short
            if action == 0 or action == 2:
                position_type = "long" if action == 0 else "short"
                if self.position.is_open():
                    if self.position.position_type != position_type:
                        profit_loss = self.close_position()
                        self.open_position(position_type, position_contracts)
                        self.current_capital += profit_loss
                else:
                    self.open_position(position_type, position_contracts)
            

    
    def get_trade_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_history)

    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02, window_size=None) -> float:
        if not self.trade_history:
            return -100

        # Extract recent trades based on window_size
        recent_trades = self.trade_history[-window_size:] if window_size else self.trade_history

        # Filter out trades with 'nan' profit_loss
        filtered_trades = [trade for trade in recent_trades if trade['profit_loss'] is not np.nan]

        if not filtered_trades:
            return -100

        # Calculate returns using list comprehension
        returns = [(trade['profit_loss'] / trade['entry_price']) if trade['type'] == 'long'
                    else (-trade['profit_loss'] / trade['entry_price']) for trade in filtered_trades]

        # Convert to NumPy array for efficient calculation
        returns_np = np.array(returns)

        mean_return = np.mean(returns_np)
        std_dev = np.std(returns_np)

        if std_dev == 0:
            return -100

        return float((mean_return - risk_free_rate) / std_dev)


    def handle_trade(self, position_type: str, position_contracts: float) -> float:
        if self.position.is_open():
            if self.position.position_type != position_type:
                profit_loss = self.close_position()
                self.open_position(position_type, position_contracts)
                return profit_loss
            return 0
        else:
            self.open_position(position_type, position_contracts)
            return 0

    def open_position(self, position_type: str, position_contracts: float):
        entry_price = self.close_prices[self.current_index]
        self.position = TradePosition(position_type, entry_price, position_contracts)
        # Append a dictionary to the trade history list
        self.trade_history.append({
            'index': self.current_index,
            'entry_price': entry_price,
            'exit_price': np.nan,
            'type': position_type,
            'contracts': position_contracts,
            'profit_loss': np.nan
        })


    def close_position(self) -> float:
        if not self.position.is_open():
            return 0

        exit_price = self.close_prices[self.current_index]
        profit_loss = self.calculate_profit_loss(exit_price)
        last_trade = self.trade_history[-1]
        last_trade['exit_price'] = exit_price
        last_trade['profit_loss'] = profit_loss
        last_trade['close_index'] = self.current_index
        self.position = TradePosition()
        return profit_loss

    def calculate_profit_loss(self, exit_price: float) -> float:
        if self.position.position_type == "long":
            return (exit_price - self.position.entry_price) * self.position.contracts
        elif self.position.position_type == "short":
            return (self.position.entry_price - exit_price) * self.position.contracts
        return 0