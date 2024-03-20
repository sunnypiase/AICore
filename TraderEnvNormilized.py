import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import pandas as pd

class TraderEnvNormalized(gym.Env):
    MAX_TIME_WITHOUT_TRADE = 1500
    RISK_FREE_RATE = 0.02
    RISK_FREE_RATE_PER_STEP = (1 + RISK_FREE_RATE) ** (1 / 525600) - 1  # Compounded for each minute

    def __init__(self, df_raw, df_normalized, trade_size_dollars=5_000, initial_capital=10_000, render_mode='human'):
        super(TraderEnvNormalized, self).__init__()
        self.render_mode = render_mode
        self.action_space = Discrete(4)  # Buy, Sell, Close, Hold
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        self.df_raw = df_raw
        self.df_normalized = df_normalized
        self.trade_size_dollars = trade_size_dollars
        self.commission = self.calculate_commission(trade_size_dollars)
        self.initial_capital = initial_capital
        
        self.reset()

    def step(self, action):
        self.current_step += 1
        self.current_price = self.df_raw['close'].iloc[self.current_step]
        profitLose = self.process_action(action, self.current_price)
        self.state = self._next_observation()
        self.sharp_ratio = self.calculate_sharpe_ratio()
        if self.current_capital < self.trade_size_dollars or  self.sharp_ratio< -20:
            self.stop = True

        reward = 1 + profitLose #self.calculate_reward()

        done = self.current_step >= len(self.df_raw) - 1 or self.stop
        info = {'current_capital': self.current_capital, 'sharp_ratio': self.sharp_ratio, 'current_step': self.current_step,
                'trades_amount': len(self.trade_history)} if done else {}
        
        return self.state, reward, done, done, info

    def calculate_commission(self, trade_size_dollars):
        commission_rate = 0.05 / 100
        return trade_size_dollars * commission_rate

    def reset(self, **kwargs):
        self.current_step = 0
        self.current_position = None
        self.position_changed = False
        self.current_capital = self.initial_capital
        self.trade_history = []
        self.current_price = self.df_raw['close'].iloc[self.current_step]
        self.stop = False
        self.sharp_ratio = 0
        self.state = self._next_observation()
        return self.state, None

    def calculate_reward(self):
            simple_return = self.trade_history[-1] if self.trade_history else 0
            unrealized_pnl = self.calculate_profit_loss(self.current_price) if self.current_position else 0
            trade_risk = self.calculate_trade_risk()
            holding_reward = self.calculate_holding_reward()

            alpha, beta, gamma = 0.5, 0.3, 0.2
            reward = alpha * simple_return + beta * unrealized_pnl - gamma * trade_risk + holding_reward
            return reward

    def calculate_trade_risk(self):
        window_size = 10
        if len(self.trade_history) < window_size:
            return 0
        recent_returns = self.trade_history[-window_size:]
        return np.std(recent_returns)

    def calculate_holding_reward(self):
        if not self.current_position or not self.position_open_step:
            return 0

        holding_duration = self.current_step - self.position_open_step
        min_holding_duration = 20  # Minimum desired holding duration
        max_holding_duration = 256  # Maximum desired holding duration

        # Penalty for holding a position for too short a time
        if holding_duration < min_holding_duration:
            return -0.01 * (min_holding_duration - holding_duration)

        # Penalty for holding a position for too long
        elif holding_duration > max_holding_duration:
            return -0.01 * (holding_duration - max_holding_duration)

        # No penalty if within the desired holding duration range
        else:
            return 0

    def render(self, mode='human', close=False):
        if self.position_changed:
            if self.current_position:
                print(f"Position Opened: Type: {self.current_position['type']}, Entry Price: {self.current_position['entry_price']}, Step: {self.current_step}")
            else:
                time_in_position = self.current_step - self.position_open_step if self.position_open_step is not None else 0
                last_trade_return = self.trade_history[-1] if self.trade_history else 0
                print(f"Position Closed: Exit Price: {self.current_price}, Close Step: {self.current_step}, Time in Position: {time_in_position}, Return from Last Trade: {last_trade_return}")
            self.position_changed = False 
        # else:
        #     print(f"Current Step: {self.current_step}, Current Price: {self.current_price}, Current Capital: {self.current_capital}, sharp_ratio: {self.calculate_sharpe_ratio()}")

    
    def _next_observation(self):
        market_observation = self.df_normalized.iloc[self.current_step]
        position_state = 1 if self.current_position and self.current_position["type"] == "long" else \
                        -1 if self.current_position and self.current_position["type"] == "short" else 0
        return np.append(market_observation, position_state)

    def process_action(self, action, current_price):
        if action == 1:  # Buy
            self.open_position("long", current_price)
            return 0
        elif action == 2:  # Sell
            self.open_position("short", current_price)
            return 0
        elif action == 3:  # Close position
            return self.close_position(current_price)
        return self.update_position_status(current_price)
    
    def open_position(self, position_type, current_price):
        if not self.current_position and self.current_capital >= self.trade_size_dollars:
            contracts = self.trade_size_dollars / current_price
            self.current_position = {"type": position_type, "entry_price": current_price, "contracts": contracts}
            self.position_changed = True
            self.position_open_step = self.current_step  # Set the opening step

    def close_position(self, current_price):
        if self.current_position:
            profit_loss = self.calculate_profit_loss(current_price)
            profit_loss -= self.commission
            self.trade_history.append(profit_loss)
            self.current_capital += profit_loss
            self.current_position = None
            self.position_changed = True
            
            return profit_loss
        return 0


    def calculate_profit_loss(self, exit_price):
        if not self.current_position:
            return 0
        contracts = self.current_position["contracts"]
        profit_loss = (exit_price - self.current_position["entry_price"]) * contracts
        return profit_loss if self.current_position["type"] == "long" else -profit_loss

    def calculate_sharpe_ratio(self):
        risk_free_rate_per_step = TraderEnvNormalized.RISK_FREE_RATE_PER_STEP
        risk_free_return_for_trade_size = risk_free_rate_per_step * self.trade_size_dollars * self.current_step

        if not self.trade_history:
            # print("\nAdjusted Total Returns: N/A, Returns Std: N/A, Risk-Free Return for Trade Size: {:.3f}, Steps: {}, Sharpe Ratio: {:.3f}".format(risk_free_return_for_trade_size, self.current_step, -risk_free_return_for_trade_size))
            return -risk_free_return_for_trade_size

        total_returns = sum(self.trade_history)
        returns_std = np.std(self.trade_history)

        if returns_std == 0:
            # print("\nAdjusted Total Returns: N/A, Returns Std: 0.000, Risk-Free Return for Trade Size: {:.3f}, Steps: {}, Sharpe Ratio: {:.3f}".format(risk_free_return_for_trade_size, self.current_step, -risk_free_return_for_trade_size))
            return -risk_free_return_for_trade_size

        adjusted_total_returns = total_returns - risk_free_return_for_trade_size
        # print("\nAdjusted Total Returns: {:.3f}, Returns Std: {:.3f}, Risk-Free Return for Trade Size: {:.3f}, Steps: {}, Sharpe Ratio: {:.3f}".format(adjusted_total_returns, returns_std, risk_free_return_for_trade_size, self.current_step, adjusted_total_returns / returns_std))
        return adjusted_total_returns / returns_std

    def update_position_status(self, current_price):
        if self.current_position:
            if self.current_position["type"] == "long" and current_price < self.current_position["entry_price"] * 0.95:
                return self.close_position(current_price)
            elif self.current_position["type"] == "short" and current_price > self.current_position["entry_price"] * 1.05:
                return self.close_position(current_price)
        return 0
