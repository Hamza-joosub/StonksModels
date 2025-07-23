from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import math
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import time
import requests
import os
#from stqdm import stqdm
from sklearn.manifold import TSNE
import plotly_express as px
import yfinance as yf
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  
API_KEY = "6ulfs8VItWZcKZTMzNJxwmikpQvSF1cI"

def Portfolio_Variance_Calculator():
    stock_list_pd = pd.read_pickle("StockList")
    Tickers = st.multiselect("Choose Stocks", options=stock_list_pd["symbol"].to_list())
    weights = []
    start = "2022-01-01"
    end = datetime.today().strftime('%Y-%m-%d')
        
    if len(Tickers) > 1:
        st.markdown("## Portfolio Correlation")
        temp_df = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={Tickers[0]}&apikey={API_KEY}").json())
        temp_df = temp_df.drop(columns=['symbol', 'volume'])
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df = temp_df.set_index('date')
        raw_data = pd.DataFrame(temp_df.index)
        raw_data = raw_data.set_index("date")
        
        for tick in Tickers:
            st.markdown(tick)
            temp_df = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={tick}&apikey={API_KEY}").json())
            temp_df = temp_df.drop(columns=['symbol', 'volume'])
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df = temp_df.set_index('date')
            raw_data[f"{tick}"] = temp_df["price"].values.tolist()
        diffirenced_data = raw_data.pct_change()
        corr_matrix = diffirenced_data.corr()
        fig = px.imshow(corr_matrix, aspect='auto', color_continuous_scale='sunsetdark')
        st.plotly_chart(fig)
        
        st.markdown("## Portfolio Variance")
        with st.form("form4"):
            for i in range(len(Tickers)):
                temp_weight = st.number_input(f'Weight for {Tickers[i]} (%):', key=i, step=5)
                weights.append(temp_weight)
            variance_weights = st.form_submit_button("Confirm")
        
        if variance_weights:
            # Ensure weights sum to 1 (100%)
            weights = [w / 100 for w in weights]  # Convert percentages to fractions
            covariance_matrix = diffirenced_data.cov()
            total_risk_Exposure = np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Display each stock's volatility
            for ticker in Tickers:
                volatility_annualized = raw_data[ticker].pct_change().std() * np.sqrt(252)  # annualized volatility
                volatility_daily = raw_data[ticker].pct_change().std()
                st.markdown(f"Annualized Volatility of {ticker}: {round(volatility_annualized * 100, 2)}%, With Daily Volatility of {ticker}: {round(volatility_daily * 100, 2)}%")
                st.markdown(f"")
            
            portfolio_variance = total_risk_Exposure
            portfolio_volatility_annaulized = math.sqrt(portfolio_variance) * np.sqrt(252)  # annualized volatility
            portfolio_volatility_daily = math.sqrt(portfolio_variance)
            portfolio_return = np.dot(diffirenced_data.mean(), weights)*10000
            
            st.markdown(f'### Annualized Portfolio Volatility: {round(portfolio_volatility_annaulized * 100, 2)}%')
            st.markdown(f'### Daily Portfolio Volatility: {round(portfolio_volatility_daily * 100, 2)}%')
            st.markdown("----")
            st.markdown("# Efficient Frontier")
            alpha = []
            stds = []
            w = []
            num_of_iterations = 130000
            
            progress_num = 0
            loading_bar = st.progress(0, text="Simulating Portfolio's")
            for i in (range(130000)):
                weights = np.random.random(len(diffirenced_data.columns))
                weights /= weights.sum()
                alpha.append((np.dot(diffirenced_data.mean(), weights)*10000))
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                stds.append((math.sqrt(portfolio_variance)*np.sqrt(252))*10)
                w.append(weights)
                progress_num = progress_num +1
                loading_bar.progress(progress_num/num_of_iterations)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = stds, y = alpha, mode='markers',marker=dict(color='blue',size=3), name="Random Portfolio's" ))
            fig.add_trace(go.Scatter(x = [portfolio_volatility_annaulized*10], y = [portfolio_return], mode='markers',marker=dict(color='green',size=15),name='Current Portfolio'))
            fig.update_xaxes(title_text = 'Portfolio Standard Deviation(%)')
            fig.update_yaxes(title_text = 'Return(%)')
            fig.update_layout(
                autosize=False,
                width=1300,
                height=500,
            )
            st.plotly_chart(fig)
            portfolio_metrics = pd.DataFrame()
            portfolio_metrics['Weights'] = w
            portfolio_metrics['returns'] = alpha
            portfolio_metrics['Vol'] = stds
            portfolio_metrics['Sharpe'] = portfolio_metrics['returns']/portfolio_metrics['Vol']
            portfolio_metrics = portfolio_metrics.sort_values('Sharpe', ascending=False)
            portfolio_weights = pd.DataFrame()
            portfolio_weights['Ticker'] = Tickers
            for i in range(0,4):
                portfolio_return = portfolio_metrics.iloc[i,1]
                portfolio_sharpe = portfolio_metrics.iloc[i,3]
                portfolio_weights[f'Portfolio Rank {i+1} With Return: {round(portfolio_return,2)}% and Sharpe: {round(portfolio_sharpe,2)}'] = portfolio_metrics.iloc[i,0].tolist() 
            st.dataframe(portfolio_weights)
            
def dcfModel():
    ticker = st.text_input(label="Enter ticker", placeholder='AAPL')
    
    company_profile = requests.get(f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={API_KEY}").json()[0]
    symbol = company_profile['symbol']
    currentPrice = company_profile['price']
    marketCap =company_profile['marketCap']
    changePercentage = company_profile['changePercentage']
    exchange = company_profile['exchange']
    industry = company_profile['industry']
    description = company_profile['description']
    sector = company_profile['sector']
    name = company_profile['companyName']
    beta = company_profile['beta']
    
    income_stmnt = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&apikey={API_KEY}").json())
    income_stmnt = income_stmnt.drop(columns=['symbol', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
    income_stmnt = income_stmnt.set_index('fiscalYear')
    income_stmnt_Metrics = income_stmnt.columns.values.tolist()
    
    
    sharesOutstanding = income_stmnt.iloc[0,-2]
    
    balance_sheet_stmnt = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&apikey={API_KEY}").json())
    balance_sheet_stmnt = balance_sheet_stmnt.drop(columns=['symbol','fiscalYear', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
    
    
    
    cash_flow_stmnt = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&apikey={API_KEY}").json())
    cash_flow_stmnt = cash_flow_stmnt.drop(columns=['symbol', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
    cash_flow_stmnt = cash_flow_stmnt.set_index('fiscalYear')
    cash_flow_stmnt_Metrics = cash_flow_stmnt.columns.values.tolist()
    
    statement = pd.concat([income_stmnt,balance_sheet_stmnt,cash_flow_stmnt], axis=1)
    statement = statement.iloc[::-1].reset_index(drop=True)
    statements_columns = statement.columns.values.tolist()
    
   

    
    metric_name = st.selectbox('Select Line Item to Plot', options=statements_columns, placeholder='Select Line Item')
    if metric_name in cash_flow_stmnt_Metrics:
        metric_no_na = cash_flow_stmnt[metric_name]
        metric_no_na = metric_no_na.dropna()
        metric_no_na = metric_no_na.iloc[::-1]
        metric_growth = metric_no_na.pct_change(fill_method=None)
        metric_growth = metric_growth.iloc[::-1]
        metric_growth=metric_growth.astype(float)*100
        metric_growth = metric_growth.round(decimals=2)
        metric_growth = metric_growth.astype('string') + " %"
        metric_list_form = cash_flow_stmnt[metric_name].dropna().to_list()
        cagr = (metric_list_form[0]/metric_list_form[-1])**(1/len(metric_list_form))-1
        if isinstance(cagr, complex):
            cagr = cagr
        else:
            cagr = round(cagr*100,2)
        meric_ot_fig = go.Figure()
        meric_ot_fig.add_traces(go.Bar(x = cash_flow_stmnt.index,y = cash_flow_stmnt[metric_name], text=metric_growth, marker_color='salmon'))
        meric_ot_fig.update_xaxes(title_text = 'Date')
        meric_ot_fig.update_yaxes(title_text = metric_name)
        meric_ot_fig.update_layout(title_text=f'{metric_name} Over Time with CAGR: {cagr} %')
    elif metric_name in income_stmnt_Metrics:   
        metric_no_na = income_stmnt[metric_name]
        metric_no_na = metric_no_na.dropna()
        metric_no_na = metric_no_na.iloc[::-1]
        metric_growth = metric_no_na.pct_change(fill_method=None)
        metric_growth = metric_growth.iloc[::-1]
        metric_growth=metric_growth.astype(float)*100
        metric_growth = metric_growth.round(decimals=2)
        metric_growth = metric_growth.astype('string') + " %"
        metric_list_form = income_stmnt[metric_name].dropna().to_list()
        cagr = (metric_list_form[0]/metric_list_form[-1])**(1/len(metric_list_form))-1
        if isinstance(cagr, complex):
            cagr = cagr
        else:
            cagr = round(cagr*100,2)
        meric_ot_fig = go.Figure()
        meric_ot_fig.add_traces(go.Bar(x = income_stmnt.index,y = income_stmnt[metric_name], text=metric_growth, marker_color='salmon'))
        meric_ot_fig.update_xaxes(title_text = 'Date')
        meric_ot_fig.update_yaxes(title_text = metric_name)
        meric_ot_fig.update_layout(title_text=f'{metric_name} Over Time With CAGR {cagr} %')
    else:
        metric_no_na = balance_sheet_stmnt[metric_name]
        metric_no_na = metric_no_na.dropna()
        metric_no_na = metric_no_na.iloc[::-1]
        metric_growth = metric_no_na.pct_change(fill_method=None)
        metric_growth = metric_growth.iloc[::-1]
        metric_growth=metric_growth.astype(float)*100
        metric_growth = metric_growth.round(decimals=2)
        metric_growth = metric_growth.astype('string') + " %"
        metric_list_form = balance_sheet_stmnt[metric_name].dropna().to_list()
        cagr = (metric_list_form[0]/metric_list_form[-1])**(1/len(metric_list_form))-1
        if isinstance(cagr, complex):
            cagr = cagr
        else:
            cagr = round(cagr*100,2)
        meric_ot_fig = go.Figure()
        meric_ot_fig.add_traces(go.Bar(x = balance_sheet_stmnt.index,y = balance_sheet_stmnt[metric_name], text=metric_growth, marker_color='salmon'))
        meric_ot_fig.update_xaxes(title_text = 'Date')
        meric_ot_fig.update_yaxes(title_text = metric_name)
        meric_ot_fig.update_layout(title_text=f'{metric_name} Over Time With CAGR: {cagr} %')
        
    st.plotly_chart(meric_ot_fig,use_container_width=True)

    st.markdown("## Future Cash Flows")
    #Ask for Forecast Length
    n = st.number_input('Forecast Length', min_value=2, step=1)
    fcf_Growth_list = []
    cagr_toggle = st.toggle("Use CAGR Of FCF Instead", value=False)
    if cagr_toggle:
        cagr_fcf = st.number_input('Input CAGR Of Free Cash Flows', step=0.5)
        g = (st.number_input(f'Terminal Growth Rate',step=0.5))/100
        future_Cash_Flows = {'Free Cash Flow Current': cash_flow_stmnt['freeCashFlow'].to_list()[0]
                }
        for i in range(1,n+1):
            future_Cash_Flows[f'Forecast {i+1}'] =cash_flow_stmnt['freeCashFlow'].to_list()[0]*(1+cagr_fcf/100)**i
          
            
    else:
        
        for i in range(1,n+1):
                temp_FCF_Growth = st.number_input(f'Free Cash Flow Growth Forecast {i}', step=0.5)
                fcf_Growth_list.append(temp_FCF_Growth)
        curr_FCF = cash_flow_stmnt['freeCashFlow'].to_list()[0]
        g = (st.number_input(f'Terminal Growth Rate', step=0.5))/100
        future_Cash_Flows = {'Free Cash Flow Current': cash_flow_stmnt['freeCashFlow'].to_list()[0]
                }
        temp_fcf = curr_FCF
        for i in range(0,len(fcf_Growth_list)):
            temp_fcf = temp_fcf+((fcf_Growth_list[i]/100)*temp_fcf)
            future_Cash_Flows[f'Forecast {i+1}'] =temp_fcf
    st.table(future_Cash_Flows)
    
    
    st.markdown("## Cost of Equity(USING CAPM)")
    #Get Risk Free Rate
    risk_free_rate = st.selectbox('Choose Risk Free Rate Proxy', options=['10 Year Treasury Yield'])
    if risk_free_rate == '3 Month Treasury Yield':
        tbill = yf.Ticker("^IRX")
    else:
        tbill = requests.get(f'https://financialmodelingprep.com/api/v4/treasury?tenor=10y&apikey={API_KEY}').json()[0]["month1"]
    
    st.markdown(tbill)
    rf = tbill/100

    
    #get Market Return
    market_index_long_name = st.selectbox('Choose Market Index', options=['S&P500'])
    annualized_over = st.selectbox('Choose How long To annualize Returns over', options=['1 Year', '5 Years', '10 Years'])
    if market_index_long_name == 'S&P500':
        spy_returns = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/^GSPC?serietype=line&apikey={API_KEY}').json()['historical'])
        spy_returns = spy_returns.iloc[0:2500]
        rm = (spy_returns.iloc[0,1]/spy_returns.iloc[-1,1])**(1/10)-1
    
        
    
    
    st.markdown(f'Risk Free Rate({risk_free_rate}): {round(rf*100,3)} %')
    st.markdown(f'Market Return({market_index_long_name}) {round(rm*100,3)} %')
    st.markdown(f"Beta: {beta}")
    rc = round((rf + beta*(rm-rf)),2)
    st.markdown(f'##### Cost Of Equity: {round(rc*100,2)} %')
    
    st.markdown("## Cost of Debt")
    if math.isnan(income_stmnt['interestExpense'].to_list()[0]):
        interest_expense = 0
    else:
        interest_expense = income_stmnt['interestExpense'].to_list()[0]
    total_debt = balance_sheet_stmnt['totalDebt'].to_list()[0]
    rd = (interest_expense/total_debt)
  
    st.markdown(f'Interest Expense: {interest_expense}')
    st.markdown(f'Total Debt: {total_debt}')
    st.markdown(f'##### Cost of Debt: {round(rd*100,2)} %')
    
    st.markdown("## Weighted Average Cost of Capital")
    total_Equity = balance_sheet_stmnt['totalEquity'].to_list()[0]
    st.markdown(f"Total Debt: {total_debt}")
    st.markdown(f"Total Equity: {total_Equity}")
    wacc = ((total_Equity/(total_Equity+total_debt))*rc)+((total_debt/(total_Equity+total_debt))*rd)
    st.markdown(f'##### Wacc: {round(wacc*100,2)}%')
    
    st.markdown('## Discounted Future Cash Flows to Present Value')
    future_cash_Flows_list = list(future_Cash_Flows.values())
    tv = (future_cash_Flows_list[-1]*(1+g))/(wacc-g)
    pv = 0
    for t in range(1,len(future_cash_Flows_list)):
        print(future_cash_Flows_list[t])
        pv = pv + future_cash_Flows_list[t]/(1+wacc)**t
    ev = pv+(tv/(1+wacc)**(len(future_cash_Flows_list)+1))
    net_debt = balance_sheet_stmnt['netDebt'].to_list()[0]
    st.markdown(f'##### Equity Value: {round(ev,0)}') 
    
    st.markdown('## Intrinsic Value per Sare')
    st.markdown(f"Net Debt: {net_debt}")
     
    iv = ((ev-net_debt)/sharesOutstanding)
    st.markdown(f"#### Intrinsic Value Per Share: ${round(iv,2)}")
    st.markdown(f"#### Current Price : ${currentPrice}")
    if currentPrice>iv+iv*0.1:
        status = 'Overvalued'
    elif currentPrice<iv-iv*0.1:
        status = 'undervalued'
    else:
        status = 'Fairly Valued'
    st.markdown(f"## {name} is {status}")
    
    st.markdown("# Summary")
    
    if len(fcf_Growth_list) == 0:
        user_prompt = f"This is a DCF Model to value {ticker}, The current free cash flow is: {cash_flow_stmnt['freeCashFlow'].to_list()[0]}, The user inputted a {cagr_fcf} Compounded annual growth rate of FCF over the next {n} years, the terminal growth rate is {g*100}%,WACC is calulcated by CAPM where the risk free rate used is {risk_free_rate}%,The market return is the return of the {market_index_long_name} annaulised over {annualized_over} and the cost of debt is calculated and thus the WACC being: {round(wacc*100,2)},The intrinsic value of the share is calculated as {iv}. Here are the previous years cash flow for {ticker}: {cash_flow_stmnt['freeCashFlow'].to_list()}. Give an in depth analysis of the Model, Provide feedback if changes of any inputs are needed"

    if st.button("Generate Response"):
        payload = {
            "model": "llama3",
            "prompt": user_prompt,
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)

            if response.status_code == 200:
                st.markdown("### 🤖 Ollama Response:")
                st.markdown(response.json()["response"])
            else:
                st.error(f"❌ API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Ollama server is not reachable. Make sure it is running.")

def check_ollama_running():
    """Check if Ollama is already running."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False
    return False

def start_ollama():
    """Start Ollama server if not running."""
    if not check_ollama_running():
        st.markdown("🔄 Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Give some time for the server to start
        if check_ollama_running():
            st.success("✅ Ollama is now running!")
        else:
            st.error("❌ Failed to start Ollama.")
    else:
        st.success("✅ Ollama is already running.")

def sector_screener():
    st.markdown("# Sector Screener")
    st.markdown("## Sector Metrics")
    df = pd.read_csv("Multiples_Database_JSE.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    #df = df.dropna()
    list_of_metrics = df.columns.to_list()
    list_of_metrics.remove('Company')
    list_of_metrics.remove('Industry')
    list_of_metrics.remove('Sector')
    list_of_metrics.remove('Name')
    sectors = df['Sector'].unique().tolist()
    sector_metrics = pd.DataFrame()
    
    for metric in list_of_metrics:
        temp_list = []
        if metric == "Market Cap":
            for sector in sectors:
                temp_list.append(df[df['Sector']==sector][metric].sum())
            sector_metrics['Total Market Cap'] = temp_list
        
        elif metric == "Revenue":
            for sector in sectors:
                temp_list.append(df[df['Sector']==sector][metric].sum())
            sector_metrics['Total Revenue'] = temp_list
            
        elif metric == "EBITDA":
            for sector in sectors:
                temp_list.append(df[df['Sector']==sector][metric].sum())
            sector_metrics['Total EBITDA'] = temp_list
        else:
            for sector in sectors:
                temp_list.append(df[df['Sector']==sector][metric].mean())
            sector_metrics[f'Average Sector: {metric}'] = temp_list
    sector_metrics["Sector"] = sectors
    sector_metrics = sector_metrics.set_index("Sector")
    
    metric_to_plot = st.selectbox("Select Metric To Plot", options=sector_metrics.columns.to_list())
    sector_fig = go.Figure()
    temp = sector_metrics.sort_values([metric_to_plot])
    sector_fig.add_trace(go.Bar(x = temp[metric_to_plot], y = temp.index, orientation='h' ))
    sector_fig.update_xaxes(title_text = f"{metric_to_plot}")
    sector_fig.update_yaxes(title_text = f"Sector")
    sector_fig.update_layout(
                autosize=False,
                width=1000,
                height=500,
            )
    st.plotly_chart(sector_fig)
    st.markdown("## Sector Ranks")
    columns_to_reverse = ['Average Sector: EV/EBITDA', 'Average Sector: EV/Rev', 'Average Sector: Debt/Equity',
                      'Average Sector: PE', 'Average Sector: Beta','Average Sector: Trailing PEG Ratio' ]
    sector_ranks = pd.DataFrame()
    for column in sector_metrics.columns:
        if column in columns_to_reverse:
            sector_ranks[column + ' Rank'] = sector_metrics[column].rank(ascending=True)  # Lower is better
        else:
            sector_ranks[column + ' Rank'] = sector_metrics[column].rank(ascending=False)  # Higher is better
    sector_ranks['Average Rank'] = sector_ranks.sum(axis=1)/(sector_ranks.shape[1]-1)
    sector_ranks['Median Rank'] = sector_ranks.median(axis=1)
    #st.dataframe(sector_ranks)
    
    # Assuming 'ranked_df' is your DataFrame with ranks
    fig = go.Figure(data=go.Heatmap(
        z=sector_ranks.values,
        x=sector_ranks.columns,
        y=sector_ranks.index,
        colorscale='thermal',  # This is a visually appealing color scale; you can choose any
        reversescale=True,  # Reverse the color scale to match the rank logic
        colorbar=dict(title='Rank')
    ))

    fig.update_layout(
        title='Sector Metric Rankings',
        xaxis_title='Metrics',
        yaxis_title='Sectors',
        yaxis_autorange='reversed'  # Optionally reverse the y-axis to have the top rank at the top
    )


    fig.update_layout(
                    autosize=False,
                    width=1300,
                    height=700,
                )

    st.plotly_chart(fig, use_container_width=True)
    
    
    
    st.markdown("## Summary")
    
    user_prompt = f'''
    You are investment Analyst lookin for potential sectors to invest in. 
    You have been given the following data for the 11 sectors of the us economy {sector_metrics.to_string()} in a HTML Format. 
    Using this data provide detailed breakdown of which sectors would be viable to invest in. Give Justifications. '''

    #st.markdown(f"Prompt: {sector_metrics.to_html()}")
    
    
    if st.button("Generate Response"):
        payload = {
            "model": "llama3",
            "prompt": user_prompt,
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)

            if response.status_code == 200:
                st.markdown("### 🤖 Ollama Response:")
                st.markdown(response.json()["response"])
            else:
                st.error(f"❌ API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Ollama server is not reachable. Make sure it is running.")
    
    st.markdown("----")
    st.markdown("## List Stocks Under Each Sector")
    sector_choice = st.selectbox("Input Sector", options=df['Sector'].unique())
    st.dataframe(df[df['Sector'] == sector_choice])
    
    
    
    st.markdown("-----")    
             
def Company_overview():
    
    ticker = st.text_input(label="Enter ticker", placeholder='Enter A Ticker')
    ok_button = st.button("OK")
    if ok_button:
        #get Basic Info
        company_profile = requests.get(f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={API_KEY}").json()[0]
        symbol = company_profile['symbol']
        price = company_profile['price']
        marketCap =company_profile['marketCap']
        changePercentage = company_profile['changePercentage']
        exchange = company_profile['exchange']
        industry = company_profile['industry']
        description = company_profile['description']
        sector = company_profile['sector']
        companyName = company_profile['companyName']
        
        st.markdown(f"# {companyName}")
        st.metric(label = f"{exchange}" , value=price, delta=f"{round(changePercentage,2)} %", label_visibility ='hidden' )
        
        # Stock Chart
        company_chart = requests.get(f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={ticker}&apikey={API_KEY}").json()
        company_chart_df = pd.DataFrame(company_chart)
        company_chart_df['date'] = pd.to_datetime(company_chart_df['date'])
        price_chart = go.Figure()
        price_chart.add_trace(go.Line(x = company_chart_df['date'], y = company_chart_df['price']))
        price_chart.update_layout(
            title=dict(
                text=f"{ticker} Chart"
            ),
            xaxis=dict(
                title=dict(
                    text="Date"
                )
            ),
            yaxis=dict(
                title=dict(
                    text=f"{ticker} Price"
                )
            ),
            legend=dict(
                title=dict(
                    text="Legend Title"
                )
            )
        )
        st.plotly_chart(price_chart,use_container_width=True)
        
        # Display Basic Info
        st.markdown(f"##### Sector: {sector}")
        st.markdown(f"##### Industry: {industry}")
        st.markdown(f"{description}")
        st.markdown(f"_____")
        
        
            
            
        # Product Segmentation
        st.markdown("## Revenue Segmentation")
        product_segmentation = requests.get(f"https://financialmodelingprep.com/stable/revenue-product-segmentation?symbol={ticker}&apikey={API_KEY}").json()
        product_segmentation_df = pd.DataFrame(product_segmentation)
        product_segmentation_df = product_segmentation_df.drop(columns=['symbol','period','reportedCurrency', 'fiscalYear'  ])

        # Normalize the dictionary into separate columns
        data_expanded = pd.json_normalize(product_segmentation_df['data'])

        # Combine with the 'date' column
        product_segmentation_df_final = pd.concat([product_segmentation_df['date'], data_expanded], axis=1)

        product_segmentation_df_final = product_segmentation_df_final.iloc[0:8,:]
        product_segmentation_df_final = product_segmentation_df_final.T.dropna()
        product_segmentation_df_final = product_segmentation_df_final.T
        product_segmentation_df_final['date'] = pd.to_datetime(product_segmentation_df_final['date'])
        df_melted = product_segmentation_df_final.melt(id_vars='date', var_name='Segment', value_name='Revenue')
        fig = px.area(df_melted,
                    x='date',
                    y='Revenue',
                    color='Segment',
                    title=f'{ticker} Revenue Segmentation Over Time',
                    labels={'Revenue': 'Revenue', 'date': 'Date'},
                    )
        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Segment',
            xaxis_title='Date',
            yaxis_title='Revenue (USD)',
            template='plotly_dark'  # Optional: use 'plotly_white' for light theme
        )
        #fig.show()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"_____")
        # Geographic Segmentation
        st.markdown("## Geographic Segmentation")
        geographic_segmentation = requests.get(f"https://financialmodelingprep.com/stable/revenue-geographic-segmentation?symbol={ticker}&apikey={API_KEY}").json()
        geographic_segmentation_df = pd.DataFrame(geographic_segmentation)
        geographic_segmentation_df = geographic_segmentation_df.drop(columns=['symbol','period','reportedCurrency', 'fiscalYear'  ])

        # Normalize the dictionary into separate columns
        data_expanded = pd.json_normalize(geographic_segmentation_df['data'])

        # Combine with the 'date' column
        geographic_segmentation_df_final = pd.concat([geographic_segmentation_df['date'], data_expanded], axis=1)

        geographic_segmentation_df_final = geographic_segmentation_df_final.iloc[0:8,:]
        geographic_segmentation_df_final = geographic_segmentation_df_final.T.dropna()
        geographic_segmentation_df_final = geographic_segmentation_df_final.T
        geographic_segmentation_df_final['date'] = pd.to_datetime(geographic_segmentation_df_final['date'])
        df_melted = geographic_segmentation_df_final.melt(id_vars='date', var_name='Segment', value_name='Revenue')
        fig = px.area(df_melted,
                    x='date',
                    y='Revenue',
                    color='Segment',
                    title=f'{ticker} Revenue Segmentation Over Time',
                    labels={'Revenue': 'Revenue', 'date': 'Date'},
                    )
        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Segment',
            xaxis_title='Date',
            yaxis_title='Revenue (USD)',
            template='plotly_dark'  # Optional: use 'plotly_white' for light theme
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"_____")
        # Get Statement Data
        income_statement = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&apikey={API_KEY}").json())
        income_statement = income_statement.drop(columns=['symbol', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
        
        balance_sheet = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&apikey={API_KEY}").json())
        balance_sheet = balance_sheet.drop(columns=['symbol','fiscalYear', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
        
        Cash_flow = pd.DataFrame(requests.get(f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&apikey={API_KEY}").json())
        Cash_flow = Cash_flow.drop(columns=['symbol','fiscalYear', 'reportedCurrency', 'cik','date','filingDate', 'acceptedDate', 'period' ])
        
        statement = pd.concat([income_statement,balance_sheet,Cash_flow], axis=1)
        
        statement = statement.iloc[::-1].reset_index(drop=True)
        
        statements_columns = statement.columns.values.tolist()
        
        #Line Plot
        things_to_plot = st.multiselect(label="Choose Line Item to Plot", options=statements_columns, key=1, default=["revenue", "costOfRevenue", "grossProfit", 'ebitda'])
        fig = go.Figure()
        for line_item in things_to_plot:
            
            fig.add_trace(go.Scatter(x = statement['fiscalYear'],y = statement[line_item],name=f"{line_item}") )

        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Line Item',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_dark',
            title = "Line Item Plot")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"_____")
        # Bar Plot
        things_to_plot = st.multiselect(label="Choose Line Item to Plot", options=statements_columns, key=2,default= ["totalAssets", "totalEquity", "totalLiabilities", 'freeCashFlow'])
        fig = go.Figure()
        for line_item in things_to_plot:
            
            fig.add_trace(go.Bar(x = statement['fiscalYear'],y = statement[line_item],name=f"{line_item}") )

        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Line Item',
            xaxis_title='Date',
            yaxis_title='Amount',
            template='plotly_dark',
        title = "Line Item Plot")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"_____")
        #news Headlines
        st.markdown("# News Headlines")
        stock_news = requests.get(f"https://financialmodelingprep.com/stable/news/stock?symbols={ticker}&apikey={API_KEY}").json()
        for article in stock_news[0:5]:
            st.markdown(f"### {article['title']}")
            st.markdown(article['text'])
             
with st.sidebar:
    selected = option_menu(
        menu_title = 'Models',
        options = [ 'Company Overview','DCF Model','Sector Screener','Portfolio Variance Calculator' ],
        orientation='vertical',
        icons = ['house', 'buildings', 'lock', 'buildings','buildings' ])
    
if selected == 'DCF Model':
    dcfModel()
    
if selected == 'Company Overview': 
    Company_overview()

if selected == 'Sector Screener':
    sector_screener()
    
if selected == 'Portfolio Variance Calculator':
    Portfolio_Variance_Calculator()   