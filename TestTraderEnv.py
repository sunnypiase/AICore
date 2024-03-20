import unittest
import numpy as np
import pandas as pd

from TraderEnv import TraderEnv
from TraderEnvNormilized import TraderEnvNormalized

class TestTraderEnv(unittest.TestCase):

    def setUp(self):
        mock_data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105]  # Add other necessary columns
        })
        self.env = TraderEnvNormalized(mock_data, mock_data, trade_size_dollars= 10_000, initial_capital= 100_000)
        self.initial_capital = self.env.initial_capital

    def test_buy_wait_wait_close(self):
        # Assuming initial capital is 10000
        expected_profit = 300  # Calculated from the scenario
        expected_final_capital = self.initial_capital + expected_profit

        # Execute the sequence of actions
        self.env.step(1)  # Buy
        self.env.step(0)  # Wait
        self.env.step(0)  # Wait
        _, _, _, _, _ = self.env.step(3)  # Close

        # Assert final capital
        self.assertEqual(self.env.current_capital, expected_final_capital)

    def test_sell_wait_wait_close(self):
        # Assuming initial capital is 10000
        expected_loss = -300  # Calculated from the scenario
        expected_final_capital = self.initial_capital + expected_loss

        # Execute the sequence of actions
        self.env.step(2)  # Sell
        self.env.step(0)  # Wait
        self.env.step(0)  # Wait
        _, _, _, _, _ = self.env.step(3)  # Close

        # Assert final capital
        self.assertEqual(self.env.current_capital, expected_final_capital)
    
    # def test_sharpe_ratio(self):
    #     self.env.step(1)  # Buy at 100
    #     self.env.step(0)  # Price goes to 102
    #     self.env.step(0)  # Price goes to 101
    #     self.env.step(2)  # Close at 103
    #     _, _, _, _, info = self.env.step(3)

    #     trade_return = (103 - 100) / 100
    #     trade_returns = [trade_return]
    #     total_returns = sum(trade_returns)

    #     risk_free_return_for_trade_size = self.env.RISK_FREE_RATE_PER_STEP * self.env.trade_size_dollars * len(trade_returns)
    #     adjusted_total_returns = total_returns - risk_free_return_for_trade_size

    #     returns_std = np.std(trade_returns)
    #     expected_sharpe_ratio = adjusted_total_returns / returns_std if returns_std != 0 else 0

    #     self.assertAlmostEqual(info['sharpe_ratio'], expected_sharpe_ratio, places=4)



if __name__ == '__main__':
    unittest.main()
