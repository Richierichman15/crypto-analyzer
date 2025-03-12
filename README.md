# Crypto Trading Bot with Short-Selling Support

A sophisticated cryptocurrency trading bot with machine learning capabilities, risk management, and support for both long and short positions.

## Features

- **Dual Strategy Support**: Implements both long-only and short-selling strategies
- **Market Trend Detection**: Automatically detects market trends to determine optimal trading strategy
- **Machine Learning Integration**: Uses ML models to predict price movements
- **Advanced Risk Management**: Includes position sizing, stop-loss, and take-profit mechanisms
- **Backtesting Engine**: Test strategies against historical data
- **Performance Metrics**: Track and analyze trading performance

## Short-Selling Strategy

The short-selling strategy branch adds the ability to profit from downward market movements:

- **Market Trend Analysis**: Detects bearish market conditions suitable for short positions
- **Short Position Management**: Opens short positions when downward trends are detected
- **Risk Controls**: Implements specialized risk management for short positions
- **Strategy Comparison**: Compares performance of long-only vs. short-selling strategies

## Usage

Run a backtest comparing both strategies:

```bash
python main.py backtest --start 20250301 --end 20250331 --symbols BTC,ETH,SOL,ADA
```

Run in legacy mode (long-only):

```bash
python main.py legacy --days 90 --symbols BTC,ETH
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- requests

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Project Structure

- `main.py`: Main entry point and trading bot implementation
- `trading/`: Core trading components
  - `simulator.py`: Trading simulator with short-selling support
  - `risk_manager.py`: Risk management system
- `data/`: Data handling modules
  - `market_data.py`: Current market data fetcher
  - `historical_data.py`: Historical price data fetcher
- `performance/`: Performance tracking
- `monitoring/`: Monitoring tools
- `analysis/`: Analysis utilities

## Future Improvements

- Implement real-time trading via exchange APIs
- Add more technical indicators
- Enhance ML model training with more data
- Implement portfolio optimization
- Add support for options and futures trading

## Disclaimer

This project is for educational purposes only. Cryptocurrency trading involves significant risk of loss. Use this software at your own risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 