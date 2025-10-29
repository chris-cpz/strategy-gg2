import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MomentumStrategy:
    def __init__(self, data, short_window=20, long_window=50, risk_free_rate=0.01):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.risk_free_rate = risk_free_rate
        self.signals = None
        self.positions = None

    def generate_signals(self):
        self.data['short_mavg'] = self.data['close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_mavg'] = self.data['close'].rolling(window=self.long_window, min_periods=1).mean()
        self.data['signal'] = 0
        self.data['signal'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] > self.data['long_mavg'][self.short_window:], 1, 0)
        self.data['positions'] = self.data['signal'].diff()
        self.signals = self.data[['close', 'short_mavg', 'long_mavg', 'signal', 'positions']]

    def backtest_strategy(self):
        self.data['strategy_returns'] = self.data['positions'] * self.data['close'].pct_change()
        self.data['cumulative_strategy_returns'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cumulative_market_returns'] = (1 + self.data['close'].pct_change()).cumprod()

    def calculate_performance_metrics(self):
        total_return = self.data['cumulative_strategy_returns'].iloc[-1] - 1
        daily_returns = self.data['strategy_returns'].dropna()
        sharpe_ratio = (daily_returns.mean() - self.risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)
        max_drawdown = (self.data['cumulative_strategy_returns'] / self.data['cumulative_strategy_returns'].cummax() - 1).min()
        return total_return, sharpe_ratio, max_drawdown

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['cumulative_strategy_returns'], label='Strategy Returns')
        plt.plot(self.data['cumulative_market_returns'], label='Market Returns')
        plt.title('Momentum Strategy vs Market Returns')
        plt.legend()
        plt.show()

def generate_sample_data(num_days=100):
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=num_days)
    prices = np.random.normal(loc=100, scale=1, size=num_days).cumsum()
    return pd.DataFrame(data={'close': prices}, index=dates)

if __name__ == "__main__":
    sample_data = generate_sample_data()
    strategy = MomentumStrategy(data=sample_data)
    strategy.generate_signals()
    strategy.backtest_strategy()
    total_return, sharpe_ratio, max_drawdown = strategy.calculate_performance_metrics()
    strategy.plot_results()
    print("Total Return: %.2f" % total_return)
    print("Sharpe Ratio: %.2f" % sharpe_ratio)
    print("Max Drawdown: %.2f" % max_drawdown)