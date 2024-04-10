import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class TraderEnvNormalized(gym.Env):

    MAX_TIME_WITHOUT_TRADE = 1500
    RISK_FREE_RATE = 0.02
    RISK_FREE_RATE_PER_STEP = (1 + RISK_FREE_RATE) ** (1 / 525600) - 1
    COMMISSION_RATE = 0.05 / 100
    LONG_POSITION = 1
    SHORT_POSITION = -1
    NO_POSITION = 0
    STOP_LOSS_THRESHOLD = 0.05  # 5% stop-loss


    def __init__(self, df_raw, df_normalized, trade_size_dollars=1_000, initial_capital=10_000, save_history=False, render_mode='human'):
        super().__init__()
        self.action_space = Discrete(4)  # Buy, Sell, Close, Hold
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(15,))
        self.render_mode = render_mode
        self.df_raw = df_raw
        self.df_normalized = df_normalized
        self.trade_size_dollars = trade_size_dollars
        self.initial_capital = initial_capital
        self.segment_start = 0
        self.segment_end = 0
        self.current_offset = 0
        self.trade_details = []
        self.save_history = save_history
        self.reset()

    def reset(self, **kwargs):
        self._segment_data()
        self.current_step = -1
        self.profit_lose= 0
        self.current_offset = self.segment_start
        self.close_prices = self.df_raw['close'].values
        self.current_price = self.close_prices[self.current_offset]
        self.current_position = None
        self.position_changed = False
        self.current_capital = self.initial_capital
        self.trade_history = []
        self.stop = False
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.commission = self.trade_size_dollars * self.COMMISSION_RATE
        self.normalized_data = self.df_normalized.values
        self.state = self._next_observation()
        
        return self.state, None
        
    def step(self, action):
        self.current_step += 1
        self.state = self._next_observation()
        actual_index = self.current_step + self.current_offset
        self.current_price = self.close_prices[actual_index]
        self._process_action(action)        
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.stop = self._check_stop_condition()

        reward = self._calculate_reward()
        if self.stop: 
            reward -= 1

        # Formatting the state for readability
        # formatted_state = ', '.join([f"{value:.6f}" for value in self.state])
        # print(f"Step: {self.current_step}")
        # print(f"Current Price: {self.current_price:.6f}")
        # print(f"Reward: {reward}")
        # print(f"Profit/Loss: {self._calculate_profit_loss():.6f}")
        # print(f"Sharpe Ratio: {self.sharpe_ratio:.6f}")
        # print(f"State: [{formatted_state}]")
        # print()
        done = actual_index == self.segment_end - 1 or self.stop
        info = self._get_info(done)
            
        return self.state, reward, done, done, info


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
        #     print(f"Current Step: {self.current_step}, Current Price: {self.current_price}, Current Capital: {self.current_capital}, sharp_ratio: {self.sharpe_ratio}")
    
    
    def _calculate_reward(self):
        reward = 1
        reward +=  self._calculate_profit_loss()/(self.trade_size_dollars * self.STOP_LOSS_THRESHOLD)
        reward += self.sharpe_ratio / 20
        return reward

    def _segment_data(self):
        # segment_size = len(self.df_raw) // 3
        # self.segment_start = np.random.randint(0, len(self.df_raw) - segment_size)
        # self.segment_end = self.segment_start + segment_size
        segment_size = len(self.df_raw)
        self.segment_start = 0
        self.segment_end = self.segment_start + segment_size

    def _next_observation(self):
        actual_index = self.current_step + self.current_offset
        market_observation = self.normalized_data[actual_index]
        position_state = self._get_position_state()

        return np.append(market_observation, [position_state, self._calculate_profit_loss()/(self.trade_size_dollars * self.STOP_LOSS_THRESHOLD)])

    def _process_action(self, action):
        if action == 1:  # Buy
            self._open_position("long")
        elif action == 2:  # Sell
            self._open_position("short")
        elif action == 3:  # Close position
            self._close_position()
        if self.current_position: # Check stop loss
            if self.current_position["type"] == "long":
                if self.current_price < self.current_position["entry_price"] * (1 - self.STOP_LOSS_THRESHOLD):
                    self._close_position()
            elif self.current_position["type"] == "short":
                if self.current_price > self.current_position["entry_price"] * (1 + self.STOP_LOSS_THRESHOLD):
                    self._close_position()
        return 0

    def _open_position(self, position_type):
        if not self.current_position and self.current_capital >= self.trade_size_dollars:
            contracts = self.trade_size_dollars / self.current_price
            self.current_position = {"type": position_type, "entry_price": self.current_price, "contracts": contracts}
            self.position_changed = True
            self.position_open_step = self.current_step
            if self.save_history:
                self.trade_details.append({
                    'step': self.current_step + self.current_offset,
                    'type': position_type,
                    'action': 'open',
                    'price': self.current_price,
                    'capital': self.current_capital
                })

    def _close_position(self):
        if self.current_position: 
            profit_loss = self._calculate_profit_loss()
            profit_loss -= self.commission
            self.trade_history.append(profit_loss)
            self.current_capital += profit_loss
            if self.save_history:
                self.trade_details.append({
                    'step': self.current_step + self.current_offset,
                    'type': self.current_position["type"],
                    'action': 'close',
                    'price': self.current_price,
                    'capital': self.current_capital
                })
            self.current_position = None
            self.position_changed = True

    def _calculate_profit_loss(self):
        if not self.current_position:
            return 0
        contracts = self.current_position["contracts"]
        profit_loss = (self.current_price - self.current_position["entry_price"]) * contracts
        return profit_loss if self.current_position["type"] == "long" else -profit_loss

    def _calculate_sharpe_ratio(self):
        if not self.trade_history:
            return 0

        total_returns = sum(self.trade_history)
        returns_std = np.std(self.trade_history)

        if returns_std == 0:
            return 0

        risk_free_return = self.RISK_FREE_RATE_PER_STEP * self.trade_size_dollars * self.current_step
        adjusted_total_returns = total_returns - risk_free_return
        return adjusted_total_returns / returns_std

    def _check_stop_condition(self):
        return self.current_capital < self.trade_size_dollars or self.sharpe_ratio < -20

    def _get_info(self, done):
        if done:
            return {
                'current_capital': self.current_capital,
                'sharpe_ratio': self.sharpe_ratio, 
                'current_step': self.current_step,
                'trades_amount': len(self.trade_history),
                'trade_details': self.trade_details
            }
        return {}

    def _get_position_state(self):
        if self.current_position:
            if self.current_position["type"] == "long":
                return self.LONG_POSITION
            elif self.current_position["type"] == "short":
                return self.SHORT_POSITION
        return self.NO_POSITION

