import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class TraderEnv(gym.Env):
    MAX_TIME_WITHOUT_TRADE = 1500
    RISK_FREE_RATE = 0.02
    RISK_FREE_RATE_PER_STEP = (1 + RISK_FREE_RATE) ** (1 / 525600) - 1  # Compounded for each minute

    def __init__(self, df, trade_size_dollars=5_000, initial_capital=10_000, render_mode='human'):
        super(TraderEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = Discrete(4)        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        self.df = df
        self.trade_size_dollars = trade_size_dollars
        self.comission = self.calculate_commission(trade_size_dollars)
        self.initial_capital = initial_capital
        self.reset_env()

    def step(self, action):
        self.current_price = self.df['close'].iloc[self.current_step]
        self.process_action(action, self.current_price)
        self.state = self._next_observation()
        if self.current_capital < self.trade_size_dollars:  # or some other minimum threshold
            # Significant negative reward for losing all capital
            punishment = -1000
            reward = punishment
            self.stop = True  # Optionally, you can stop the environment here
        else:
            # Regular reward calculation
            reward = self.calculate_sharpe_ratio()

        done = self.current_step >= len(self.df) - 1 or self.stop
        info = {'current_capital': self.current_capital, 'sharpe_ratio': reward, 'current_step' : self.current_step, 'trades_amount': len(self.trade_history)} if done else {}

        self.current_step += 1
        return self.state, reward, done, done, info
    
    def calculate_commission(self, trade_size_dollars):
        commission_rate = 0.05 / 100
        commission = trade_size_dollars * commission_rate
        return commission
    
    def reset(self, **kwargs):
        self.current_step = 0
        self.current_position = None
        self.current_capital = self.initial_capital
        self.trade_history = []        
        self.current_price = self.df['close'].iloc[self.current_step]
        self.stop = False
        # Create and return the initial observation
        self.state = self._next_observation()
        return self.state, None

    def render(self, mode='human', close=False):
        position_state = self.current_position['type'] if self.current_position else "Wait"
        entry_price = self.current_position['entry_price'] if self.current_position else 0

        print(f"Step: {self.current_step:<5} | Position State: {position_state:<6} | Entry Price: {entry_price:10.3f} | Current Price: {self.current_price:10.3f} | Capital: {self.current_capital:10.3f} | Sharp Ratio: {self.calculate_sharpe_ratio():10.3f}")

    def reset_env(self):
        self.current_step = 0
        self.current_position = None
        self.current_capital = self.initial_capital
        self.trade_history = []
        self.stop = False
        self.state = self._next_observation()
        return self.state
    
    def _next_observation(self):
        market_observation = self.df.iloc[self.current_step]

        # Simplified trade position state (1: open-long, -1: open-short, 0: closed)
        position_state = 1 if self.current_position and self.current_position["type"] == "long" else \
                        -1 if self.current_position and self.current_position["type"] == "short" else 0

        # Combine market data and position state
        return np.append(market_observation, position_state)

    def process_action(self, action, current_price):
        if action == 1:  # Buy
            self.open_position("long", current_price)
        elif action == 2:  # Sell
            self.open_position("short", current_price)
        elif action == 3:  # Close position
            self.close_position(current_price)
        self.update_position_status(current_price)

    def open_position(self, position_type, current_price):
        if not self.current_position and self.current_capital >= self.trade_size_dollars:
            contracts = self.trade_size_dollars / current_price
            self.current_position = {"type": position_type, "entry_price": current_price, "contracts": contracts}

    def close_position(self, current_price):
        if self.current_position:
            profit_loss = self.calculate_profit_loss(current_price)
            profit_loss -= self.comission
            self.trade_history.append(profit_loss)
            self.current_capital += profit_loss
            self.current_position = None

    def calculate_profit_loss(self, exit_price):
        if not self.current_position:
            return 0
        contracts = self.current_position["contracts"]
        profit_loss = (exit_price - self.current_position["entry_price"]) * contracts
        return profit_loss if self.current_position["type"] == "long" else -profit_loss

    def calculate_sharpe_ratio(self):
        risk_free_rate_per_step = TraderEnv.RISK_FREE_RATE_PER_STEP
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
                self.close_position(current_price)
            elif self.current_position["type"] == "short" and current_price > self.current_position["entry_price"] * 1.05:
                self.close_position(current_price)
