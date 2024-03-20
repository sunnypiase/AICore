from re import S
import pandas as pd
import numpy as np
from typing import List, Tuple

class TradePosition:
    def __init__(self, entry_index: int = -1, position_type: str = "", entry_price: float = 0, contracts: float = 0, stop_loss_percent: float = 0.05):
        self.entry_index = entry_index
        self.position_type = position_type
        self.entry_price = entry_price
        self.contracts = contracts
        self.stop_loss_percent = stop_loss_percent

    def is_open(self) -> bool:
        return self.position_type in ["long", "short"]

    def check_stop_loss(self, current_price: float) -> bool:
        if not self.is_open():
            return False
        if self.position_type == "long":
            return current_price <= self.entry_price * (1 - self.stop_loss_percent)
        elif self.position_type == "short":
            return current_price >= self.entry_price * (1 + self.stop_loss_percent)
        return False

class Trader:
    MAX_TIME_WITHOUT_TRADE = 1500

    def __init__(self, df: pd.DataFrame, trade_size_dollars: float, initial_capital: float) -> None:
        self.close_prices = df['close'].values
        self.trade_size_dollars = trade_size_dollars
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = TradePosition()
        self.current_index = 0
        self.contract_sizes = (self.trade_size_dollars / df['close']).values  # Vectorized calculation
        self.trade_history = []
        self.time_since_last_position = 0
        self.stop = False

    def trade(self, actions: List[int]) -> None:
        for i, action in enumerate(actions):
            self.current_index = i
            current_price = self.close_prices[i]

            if self.position.is_open():
                self.handle_open_position(current_price)

            # Check for stopping conditions
            if (self.position.is_open() and self.current_index - self.position.entry_index > Trader.MAX_TIME_WITHOUT_TRADE) or \
               (self.current_capital < self.trade_size_dollars):
                self.stop = True

            if self.stop:
                break

            if not self.position.is_open():
                self.time_since_last_position += 1

            position_contracts = self.contract_sizes[i]
            self.process_action(action, position_contracts, current_price)


    def process_action(self, action: int, position_contracts: float, current_price: float):
        if action in [0, 1]:
            self.handle_entry_action(action, position_contracts)
        elif action == 2:
            self.close_position("action", current_price)

    def handle_entry_action(self, action: int, position_contracts: float):
        position_type = "long" if action == 0 else "short"
        if not self.position.is_open():
            self.open_position(position_type, position_contracts)

    def handle_open_position(self, current_price: float):
        if self.position.check_stop_loss(current_price):
            self.close_position("stop-loss", current_price)

    def get_trade_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_history)

    def calculate_sharpe_ratio(self, risk_free_rate=0.02, window_size=None) -> float:

        if not self.trade_history:
            return -100000

        recent_trades = self.trade_history[-window_size:] if window_size else self.trade_history
        filtered_trades = [trade for trade in recent_trades if trade['profit_loss'] is not np.nan]

        if not filtered_trades:
            return -100000

        returns = [(trade['profit_loss'] / trade['entry_price']) if trade['type'] == 'long'
                   else (-trade['profit_loss'] / trade['entry_price']) for trade in filtered_trades]
        returns_np = np.array(returns)

        mean_return = np.mean(returns_np)
        std_dev = np.std(returns_np)

        if std_dev == 0:
            return -100000

        return float((mean_return - risk_free_rate) / std_dev)
    
    def open_position(self, position_type: str, position_contracts: float):
        self.time_since_last_position = 0
        entry_price = self.close_prices[self.current_index]
        self.position = TradePosition(self.current_index, position_type, entry_price, position_contracts)

        self.trade_history.append({
            'index': self.current_index,
            'entry_price': entry_price,
            'exit_price': np.nan,
            'type': position_type,
            'contracts': position_contracts,
            'profit_loss': np.nan
        })

    def close_position(self, close_reason: str, current_price: float) -> float:
        if not self.position.is_open():
            return 0

        # if self.current_index - self.position.entry_index < 3:
        #     self.stop = True

        profit_loss = self.calculate_profit_loss(current_price)
        last_trade = self.trade_history[-1]

        candles_in_position = self.current_index - last_trade['index']
        last_trade.update({
            'exit_price': current_price,
            'profit_loss': profit_loss,
            'close_index': self.current_index,
            'close_reason': close_reason,
            'candles_in_position': candles_in_position
        })

        self.position = TradePosition()
        return profit_loss

    def calculate_profit_loss(self, exit_price: float) -> float:
        if self.position.position_type == "long":
            return (exit_price - self.position.entry_price) * self.position.contracts
        elif self.position.position_type == "short":
            return (self.position.entry_price - exit_price) * self.position.contracts
        return 0