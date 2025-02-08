import pandas as pd
import yfinance as yf
from tqdm import tqdm

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(url)[0]  # First table contains S&P 500 tickers

# Keep only the columns of interest
sp500_table = sp500_table[["Symbol"]]
sp500_list = sp500_table['Symbol'].to_list()

data = []



for ticker in tqdm(sp500_list):
    stock = yf.Ticker(ticker)
    
    # Extract relevant financial metrics
    try:
        market_cap = stock.info["marketCap"]
        revenue_growth = stock.info.get("revenueGrowth", None)
        earningsGrowth = stock.info.get("earningsGrowth", None)
        enterpriseToEbitda = stock.info.get("earningsGrowth", None)
        enterpriseToRevenue  = stock.info.get("earningsGrowth", None)
        ebitda_margin = stock.info.get("ebitdaMargins", None)
        operatingMargins = stock.info.get("operatingMargins", None)
        de = stock.info.get("debtToEquity", None)
        trailling_pe = stock.info.get("trailingPE", None)
        roe = stock.info.get("returnOnEquity", None)
        roa = stock.info.get("returnOnAssets", None)
        industry = stock.info.get("industryKey", None)
        sector = stock.info.get("sectorKey", None)
        longname = stock.info.get("longName", None)
        beta = stock.info.get("beta", None)
        trailingPegRatio = stock.info.get("trailingPegRatio", None)
        
        
        data.append([ticker,market_cap,revenue_growth,earningsGrowth,enterpriseToEbitda,enterpriseToRevenue,ebitda_margin,operatingMargins,de, trailling_pe, roe,roa,industry,sector, longname,beta,trailingPegRatio ])
        
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
df = pd.DataFrame(data, columns=['Company', 'Market Cap', 'Rev Growth', 'NI Growth', 'EV/EBITDA', 'EV/Rev', 'EBITDA Margin', 'Operating Margin', 'Debt/Equity', 'PE', 'ROE','ROA', 'Industry', 'Sector', 'Name', 'Beta','Trailing PEG Ratio' ])
df.to_csv("Multiples_Database")