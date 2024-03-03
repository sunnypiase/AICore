import unittest
import pandas as pd
from Trader import Trader  # Replace with your actual module name

class TestTrader(unittest.TestCase):

    def setUp(self):
        data = {'close': [100, 105, 110, 105, 100, 95, 90, 95, 100, 105]}
        self.df = pd.DataFrame(data)
        self.trade_size_dollars = 1000
        self.initial_capital = 10000
        self.trader = Trader(self.df, self.trade_size_dollars, self.initial_capital)

    def test_initial_state(self):
        self.assertEqual(self.trader.current_capital, self.initial_capital)
        self.assertFalse(self.trader.position.is_open())

    def test_long_position_open_close(self):
        # Test opening and closing a long position
        self.trader.step(0)  # Open long
        self.assertTrue(self.trader.position.is_open())
        self.assertEqual(self.trader.position.position_type, "long")

        self.trader.step(2)  # Close long and open short
        self.assertTrue(self.trader.position.is_open())
        self.assertEqual(self.trader.position.position_type, "short")

    def test_short_position_open_close(self):
        # Test opening and closing a short position
        self.trader.step(2)  # Open short
        self.assertTrue(self.trader.position.is_open())
        self.assertEqual(self.trader.position.position_type, "short")

        self.trader.step(0)  # Close short and open long
        self.assertTrue(self.trader.position.is_open())
        self.assertEqual(self.trader.position.position_type, "long")

    def test_wait_action(self):
        # Test wait action
        self.trader.step(1)  # Wait
        self.assertFalse(self.trader.position.is_open())

    def test_trade_history(self):
        # Test the trade history
        actions = [0, 1, 2, 1, 0, 1, 1, 1, 1, 1 ]
        for action in actions:
            self.trader.step(action)

        trade_history_df = self.trader.get_trade_history_df()
        self.assertEqual(len(trade_history_df), 3)  # 3 trades should have been recorded

    def test_capital_change(self):
        # Test capital change after trades
        actions = [0, 0, 2, 1, 1, 1, 1, 1, 1, 1]  # Open long, close and open short, close
        for action in actions:
            self.trader.step(action)

        self.assertEqual(self.trader.current_capital, self.trader.initial_capital+(self.df['close'][2] - self.df['close'][0]) * (self.trade_size_dollars / self.df['close'][0]))

if __name__ == '__main__':
    unittest.main()
