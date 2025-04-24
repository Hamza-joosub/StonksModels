import yfinance as yf

ticker = yf.Ticker("AAPL")
print(ticker.history(period="1d"))


