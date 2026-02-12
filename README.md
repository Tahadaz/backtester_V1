# 📈 Technical Analysis Backtester

A professional-grade backtesting application for technical analysis strategies on Moroccan market data (BMCE/MASI format).

## 🌟 Features

- **Data Layer**: Support for BMCE CSV/XLSX format with automatic normalization
- **Indicators**: Extensible indicator engine with caching (SMA, EMA, Returns)
- **Strategy**: Moving Average Crossover strategy with customizable parameters
- **Portfolio**: Full portfolio simulation with transaction costs and constraints
- **Results**: Comprehensive performance analytics and visualizations

## 🏗️ Architecture

```
Data → Indicators → Strategy → Portfolio → Results
```

### Components

1. **data.py**: Market data loading and normalization
2. **indicators.py**: Technical indicator computation with caching
3. **strategy.py**: Strategy signal generation
4. **portfolio.py**: Portfolio simulation and execution
5. **results.py**: Performance analysis and reporting
6. **app.py**: Streamlit web interface

## 🚀 Quick Start

### Local Development

1. **Clone the repository** (or navigate to the folder)
   ```bash
   cd backtester
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run backtester/app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 📊 Usage

1. **Upload Data**: Upload your BMCE format CSV or XLSX file
2. **Configure Strategy**: Set moving average parameters (Fast/Slow SMA)
3. **Portfolio Settings**: Configure initial capital, rebalance policy, and constraints
4. **Transaction Costs**: Optionally enable realistic transaction costs
5. **Run Backtest**: Click the "Run Backtest" button
6. **Analyze Results**: View performance metrics, charts, and trade history

## 📁 Data Format

The app expects BMCE-style CSV/XLSX files with the following columns:

- `Date`: Trading date
- `Ouvt`: Opening price
- `+Haut`: High price
- `+Bas`: Low price
- `Clôture`: Closing price
- `Volume`: Trading volume (optional)

## 🎯 Strategy Parameters

### Moving Average Crossover

- **Fast SMA**: Short-term moving average window (e.g., 20 days)
- **Slow SMA**: Long-term moving average window (e.g., 50 days)
- **Allow Short**: Enable short positions when fast MA < slow MA
- **NaN Policy**: How to handle warmup period (flat or nan)

## 💼 Portfolio Configuration

- **Initial Cash**: Starting capital
- **Rebalance Policy**: 
  - `on_change`: Only rebalance when signals change
  - `every_bar`: Rebalance every period
- **Max Gross Exposure**: Maximum total exposure (1.0 = 100%)
- **Cash Buffer**: Percentage of cash to keep in reserve

## 💰 Transaction Costs

Realistic Moroccan market transaction costs:

- **Brokerage**: 0.60% (60 bps) HT
- **Exchange Fee**: 0.10% (10 bps) HT
- **Settlement**: 0.20% (20 bps) HT
- **VAT**: 10% on commissions
- **Slippage**: Configurable (default 0)

## 📈 Performance Metrics

The app calculates comprehensive performance metrics:

### Returns
- Total Return
- CAGR (Compound Annual Growth Rate)
- Annual Volatility

### Risk-Adjusted
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Risk
- Maximum Drawdown
- Drawdown Duration

### Trading
- Number of Trades
- Turnover
- Total Transaction Costs

### Exposure
- Average Gross Exposure
- Average Net Exposure
- Time in Market

## 🌐 Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `backtester/app.py` file
   - Deploy!

### Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backtester/ ./backtester/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "backtester/app.py"]
```

Build and run:
```bash
docker build -t backtester .
docker run -p 8501:8501 backtester
```

## 🔧 Advanced Usage

### Running Backtests Programmatically

```python
from backtester.data import BMCEDataSource
from backtester.indicators import IndicatorEngine
from backtester.strategy import MovingAverageCrossStrategy, MovingAverageCrossParams
from backtester.portfolio import PortfolioEngine, PortfolioConfig, CostModel
from backtester.results import ResultsAnalyzer

# Load data
ds = BMCEDataSource(timezone="UTC")
md = ds.load(symbols=["IAM"], paths={"IAM": "data.xlsx"})

# Compute indicators
eng = IndicatorEngine()
specs = [...]  # Define your indicator specs
feats = eng.compute(md, specs, symbols=["IAM"])

# Generate signals
params = MovingAverageCrossParams(fast_window=20, slow_window=50)
strat = MovingAverageCrossStrategy(params)
sf = strat.generate_signals(md, feats, symbols=["IAM"])

# Run portfolio
cfg = PortfolioConfig(initial_cash=1_000_000)
port = PortfolioEngine(cfg)
res = port.run(md, sf, symbols=["IAM"])

# Analyze results
analyzer = ResultsAnalyzer()
report = analyzer.analyze(res, market_data=md, symbols=["IAM"])
print(report.metrics)
```

## 📝 Project Structure

```
backtester/
├── backtester/
│   ├── app.py              # Streamlit web interface
│   ├── data.py             # Data loading and normalization
│   ├── indicators.py       # Technical indicators
│   ├── strategy.py         # Trading strategies
│   ├── portfolio.py        # Portfolio simulation
│   ├── results.py          # Performance analysis
│   └── run.py              # CLI example
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Inspired by professional backtesting frameworks like Backtrader and Zipline
- Designed for Moroccan market data (BMCE/Casablanca Stock Exchange)

## 📧 Support

For questions or issues, please open an issue on GitHub.

---

**Happy Backtesting! 📈**
