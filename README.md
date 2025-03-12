# Crypto Trading Bot

A backtesting framework for cryptocurrency trading strategies with machine learning integration.

## Overview

This project implements a cryptocurrency trading bot with the following features:

- **Data Collection**: Fetches historical crypto data from CoinGecko API
- **Technical Analysis**: Calculates indicators like RSI, MACD, Moving Averages
- **ML Integration**: Uses XGBoost to enhance trading predictions
- **Backtesting**: Tests strategies on historical data
- **Risk Management**: Implements position sizing and risk controls
- **Performance Tracking**: Tracks and analyzes trading performance

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-analyzer.git
cd crypto-analyzer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Bot

Run a backtest with default settings:
```bash
python main.py
```

### Command-Line Options

The bot accepts several command-line arguments:

- `--balance`: Initial balance for backtesting (default: 1000.0)
- `--days`: Number of days for backtest (default: 90)
- `--symbols`: Comma-separated list of crypto symbols (default: BTC,ETH,XRP,ADA,SOL,DOT,AVAX,MATIC)
- `--cap-min`: Minimum market cap in USD (default: 5,000,000)
- `--cap-max`: Maximum market cap in USD (default: 500,000,000)

Example:
```bash
python main.py --balance 2000 --days 120 --symbols BTC,ETH,SOL,DOT
```

## Project Structure

- `main.py`: Main program and backtest orchestration
- `data/`: Data fetching and processing modules
  - `data_fetcher.py`: Fetches market data
  - `historical_data.py`: Retrieves historical price data
- `trading/`: Trading-related components
  - `simulator.py`: Simulates trading with a given balance
  - `risk_manager.py`: Handles risk management for trades
- `analysis/`: Analysis tools
  - `performance.py`: Tracks trading performance
  - `optimizer.py`: Optimizes trading strategies
- `monitoring/`: Monitoring tools
  - `monitor.py`: Monitors trading activity

## Trading Strategy

The current implementation uses a combination of:
- Technical indicators (RSI, MACD, SMA)
- Price and volume movements
- Machine learning predictions (when enough data is available)

Buy signals are generated when multiple indicators align, with trade sizing proportional to conviction.

The ML component learns from historical trades to gradually improve prediction accuracy.

## Future Improvements

1. **Enhanced ML Models**:
   - Deep learning integration
   - Sentiment analysis from news/social media
   - Reinforcement learning

2. **Additional Features**:
   - More technical indicators
   - Market regime detection
   - Correlation analysis between cryptos

3. **Improved Risk Management**:
   - Dynamic position sizing based on volatility
   - Portfolio optimization
   - Drawdown control

4. **Live Trading**:
   - Integration with exchange APIs
   - Real-time alerts
   - Trading dashboard

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading involves significant risk of loss. Use this software at your own risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 