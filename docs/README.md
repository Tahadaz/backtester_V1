# Backtest Results Website - Quick Start Guide

## ✅ Fixed Issues

1. **Plot File Naming**: Now correctly handles plot files named like `bollinger_price_panel.html`, `rsi_price_panel.html`, etc.
2. **Ledger Loading**: Properly loads trade ledgers from `assets/results/stocks/{ticker}/ledgers/{strategy}_trade_ledger.csv`
3. **Debug Logging**: Added console logs to help troubleshoot any issues

## 📁 Required Folder Structure

```
your_project/
├── index.html
├── style.css
├── app.js
├── glossary.html
└── assets/
    └── results/
        ├── manifest.json
        └── stocks/
            └── {ticker}/           (e.g., "jet")
                ├── leaderboard.csv
                ├── profile.json
                ├── plots/
                │   ├── plots.json
                │   ├── bollinger_price_panel.html
                │   ├── rsi_price_panel.html
                │   └── macd_price_panel.html
                └── ledgers/
                    ├── bollinger_trade_ledger.csv
                    ├── rsi_trade_ledger.csv
                    └── macd_trade_ledger.csv
```

## 🚀 How to Run

### Option 1: Double-Click (Simplest)
1. Place all 4 HTML/CSS/JS files in the same folder as your `assets` folder
2. Double-click `index.html`

### Option 2: Local Server (Recommended)

**Python:**
```bash
cd path/to/your/project
python -m http.server 8000
# Open: http://localhost:8000
```

**Node.js:**
```bash
npm install -g http-server
cd path/to/your/project
http-server -p 8000
# Open: http://localhost:8000
```

**VS Code:**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

## 🐛 Troubleshooting

### Check Browser Console
Open browser developer tools (F12) and check the Console tab for:
- `Loading plot: assets/results/stocks/jet/plots/bollinger_price_panel.html`
- `Extracted strategy key from "bollinger_price_panel.html": "bollinger"`
- `Attempting to load ledger from: assets/results/stocks/jet/ledgers/bollinger_trade_ledger.csv`

### Common Issues

**1. "document is not defined" error:**
- Don't run `app.js` with Node.js
- Open `index.html` in a web browser instead

**2. Plot not showing:**
- Check that plot files exist in `assets/results/stocks/{ticker}/plots/`
- Check that `plots.json` lists all your plot files

**3. Ledger not showing:**
- This is OK if the file doesn't exist
- Check console for "Ledger not found (this is OK)" message
- If you want ledgers, ensure files exist in `assets/results/stocks/{ticker}/ledgers/`

**4. Parameters not showing:**
- Check that strategy names in `leaderboard.csv` match plot filenames
- Example: Plot `bollinger_price_panel.html` should have strategy `bollinger (best)` in CSV

## 📊 Data Format Examples

### plots.json
```json
{
  "plots": [
    "bollinger_price_panel.html",
    "rsi_price_panel.html",
    "macd_price_panel.html"
  ]
}
```

### Strategy Name Matching
- Plot file: `bollinger_price_panel.html`
- Extracted strategy: `bollinger`
- Leaderboard CSV should have: `bollinger (best)` in Strategy column
- Ledger file: `bollinger_trade_ledger.csv`

## 🎨 Features

✅ Signal aggregation (majority vote + weighted vote)
✅ Color-coded signals (green=BUY, red=SELL, grey=HOLD)
✅ Signal badges in stock list
✅ Parameters displayed under each plot
✅ Trade ledger table
✅ Clickable metrics linking to glossary
✅ Horizontal plot selection buttons
✅ Professional institutional styling

## 💡 Tips

- Open browser console (F12) to see debug logs
- All metrics in the leaderboard link to glossary definitions
- Adjust weights in the Weighted Vote section to see how it changes the final signal
- If a ledger doesn't exist for a strategy, the panel just won't show (this is normal)

---

**Need Help?** Check the browser console for error messages and debug logs.
