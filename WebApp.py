from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import math
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import time
import requests
import os
from stqdm import stqdm
from sklearn.manifold import TSNE
import plotly_express as px
st.set_page_config(layout="wide")
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"


    

def risk_analysis():
    stock_list_pd = pd.read_pickle("StockList")
    Tickers = st.multiselect("Choose Stocks", options=stock_list_pd["symbol"].to_list())
    weights = []
    start = "2022-01-01"
    end = datetime.today().strftime('%Y-%m-%d')
        
    if len(Tickers) > 1:
        st.markdown("## Portfolio Correlation")
        rawData = yf.download(tickers=Tickers, start=start, end=end, interval='1d')["Close"]
        diffirenced_data = rawData.pct_change()
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
                volatility_annualized = rawData[ticker].pct_change().std() * np.sqrt(252)  # annualized volatility
                volatility_daily = rawData[ticker].pct_change().std()
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

            for i in (range(5000)):
                weights = np.random.random(len(diffirenced_data.columns))
                weights /= weights.sum()
                alpha.append((np.dot(diffirenced_data.mean(), weights)*10000))
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                stds.append((math.sqrt(portfolio_variance)*np.sqrt(252))*10)
                w.append(weights)
            
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
            st.dataframe(portfolio_metrics)

def dcfModel():
    #Get Statement and Ticker Data
    stock_list_pd = pd.read_pickle("StockList") #must Get from Database
    st.markdown('# Enter Stock Name')
    tickername = st.selectbox("Input Stock Ticker", options=stock_list_pd["symbol"].to_list(), index = stock_list_pd["symbol"].to_list().index("AAPL") )
    ticker = yf.Ticker(tickername)
    cash_flow_stmnt = ((ticker.cash_flow).T)
    income_stmnt = (ticker.income_stmt).T
    balance_sheet_stmnt = (ticker.balance_sheet).T
    info = ticker.info
    
    # Assign Info To Variables
    industry = info.get('industry')
    sector = info.get('sector')
    description = info.get('longBusinessSummary')
    beta = info.get('beta')
    sharesOutstanding =  info.get('sharesOutstanding')
    name = info.get('longName')
    currentPrice = info.get('currentPrice')
    targetHighPrice = info.get('targetHighPrice')
    targetLowPrice = info.get('targetLowPrice')
    targetMeanPrice = info.get('targetMeanPrice')
    
    # Display Basic Info
    
    st.markdown(f'# {name}')
    prices_df = ticker.history(start ='2020-01-01', auto_adjust=False)['Close']
    prices_df = prices_df.reset_index()
    stock_price_fig = go.Figure()
    stock_price_fig.add_trace(go.Scatter(x = prices_df['Date'], y=prices_df['Close']))
    stock_price_fig.update_xaxes(title_text = 'Date')
    stock_price_fig.update_yaxes(title_text = 'Price')
    stock_price_fig.add_hline(y = float(targetMeanPrice), line_color='orange', annotation_text=f'Mean target Price: {targetMeanPrice}',line_dash = 'dot')
    stock_price_fig.add_hline(y = float(targetLowPrice), line_color='red', annotation_text=f'Low target Price: {targetLowPrice}',line_dash = 'dot')
    stock_price_fig.add_hline(y = float(targetHighPrice), line_color='green', annotation_text=f'High target Price: {targetHighPrice}', line_dash = 'dot')
    st.plotly_chart(stock_price_fig)
    
    st.markdown(f"{description}")
    st.markdown(f'industry: {industry}, sector: {sector}')
    
    #Plot Line Items
    st.markdown('### Plot Line Items')
    cash_flow_stmnt_Metrics = cash_flow_stmnt.columns.to_list()
    income_stmnt_Metrics = income_stmnt.columns.to_list()
    balance_sheet_stmnt_Metrics = balance_sheet_stmnt.columns.to_list()
    all_metrics = cash_flow_stmnt_Metrics+income_stmnt_Metrics+balance_sheet_stmnt_Metrics
    st.dataframe(income_stmnt)
    
    metric_name = st.selectbox('Select Line Item to Plot', options=all_metrics, placeholder='Select Line Item')
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
        
    st.plotly_chart(meric_ot_fig)

    st.markdown("## Future Cash Flows")
    #Ask for Forecast Length
    n = st.number_input('Forecast Length', min_value=2, step=1)
    fcf_Growth_list = []
    cagr_toggle = st.toggle("Use CAGR Of FCF Instead", value=False)
    if cagr_toggle:
        cagr_fcf = st.number_input('Input CAGR Of Free Cash Flows', step=0.5)
        g = (st.number_input(f'Terminal Growth Rate',step=0.5))/100
        future_Cash_Flows = {'Free Cash Flow Current': cash_flow_stmnt['Free Cash Flow'].to_list()[0]
                }
        for i in range(1,n+1):
            future_Cash_Flows[f'Forecast {i+1}'] =cash_flow_stmnt['Free Cash Flow'].to_list()[0]*(1+cagr_fcf/100)**i
          
            
    else:
        
        for i in range(1,n+1):
                temp_FCF_Growth = st.number_input(f'Free Cash Flow Growth Forecast {i}', step=0.5)
                fcf_Growth_list.append(temp_FCF_Growth)
        curr_FCF = cash_flow_stmnt['Free Cash Flow'].to_list()[0]
        g = (st.number_input(f'Terminal Growth Rate', step=0.5))/100
        future_Cash_Flows = {'Free Cash Flow Current': cash_flow_stmnt['Free Cash Flow'].to_list()[0]
                }
        temp_fcf = curr_FCF
        for i in range(0,len(fcf_Growth_list)):
            temp_fcf = temp_fcf+((fcf_Growth_list[i]/100)*temp_fcf)
            future_Cash_Flows[f'Forecast {i+1}'] =temp_fcf
    st.table(future_Cash_Flows)
    
    
    st.markdown("## Cost of Equity(USING CAPM)")
    #Get Risk Free Rate
    risk_free_rate = st.selectbox('Choose Risk Free Rate Proxy', options=['3 Month Treasury Yield', '10 Year Treasury Yield'])
    if risk_free_rate == '3 Month Treasury Yield':
        tbill = yf.Ticker("^IRX")
    else:
        tbill = yf.Ticker("^TNX")    
    st.markdown(tbill)
    rf = tbill.info.get('previousClose')/100

    
    #get Market Return
    market_index_long_name = st.selectbox('Choose Market Index', options=['S&P500', 'JSE Top 40'])
    annualized_over = st.selectbox('Choose How long To annualize Returns over', options=['1 Year', '5 Years', '10 Years'])
    if market_index_long_name == 'S&P500':
        market_index = '^SP500TR'
    elif market_index_long_name == 'JSE Top 40':
        market_index = '^J200.JO'
        
    if annualized_over == '1 Year':
        market_data = yf.Ticker(market_index).history(period="1y")
        rm = (market_data["Close"].iloc[-1] / market_data["Close"].iloc[0]) - 1
    elif annualized_over == '5 Years':
        market_data = yf.Ticker(market_index).history(period="5y")
        rm = (market_data["Close"].iloc[-1] / market_data["Close"].iloc[0]) ** (1/5) - 1
    else:
        market_data = yf.Ticker(market_index).history(period="10y")
        rm = (market_data["Close"].iloc[-1] / market_data["Close"].iloc[0]) ** (1/10) - 1
    
    st.markdown(f'Risk Free Rate({risk_free_rate}): {round(rf*100,3)} %')
    st.markdown(f'Market Return({market_index_long_name}) {round(rm*100,3)} %')
    st.markdown(f"Beta: {beta}")
    rc = round((rf + beta*(rm-rf)),2)
    st.markdown(f'##### Cost Of Equity: {round(rc*100,2)} %')
    
    st.markdown("## Cost of Debt")
    if math.isnan(income_stmnt['Interest Expense'].to_list()[0]):
        interest_expense = 0
    else:
        interest_expense = income_stmnt['Interest Expense'].to_list()[0]
    total_debt = balance_sheet_stmnt['Total Debt'].to_list()[0]
    rd = (interest_expense/total_debt)
  
    st.markdown(f'Interest Expense: {interest_expense}')
    st.markdown(f'Total Debt: {total_debt}')
    st.markdown(f'##### Cost of Debt: {round(rd*100,2)} %')
    
    st.markdown("## Weighted Average Cost of Capital")
    total_Equity = balance_sheet_stmnt['Total Equity Gross Minority Interest'].to_list()[0]
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
    net_debt = balance_sheet_stmnt['Net Debt'].to_list()[0]
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
        user_prompt = f"This is a DCF Model to value {tickername}, The current free cash flow is: {cash_flow_stmnt['Free Cash Flow'].to_list()[0]}, The user inputted a {cagr_fcf} Compounded annual growth rate of FCF over the next {n} years, the terminal growth rate is {g*100}%,WACC is calulcated by CAPM where the risk free rate used is {risk_free_rate}%,The market return is the return of the {market_index_long_name} annaulised over {annualized_over} and the cost of debt is calculated and thus the WACC being: {round(wacc*100,2)},The intrinsic value of the share is calculated as {iv}. Here are the previous years cash flow for {tickername}: {cash_flow_stmnt['Free Cash Flow'].to_list()}. Give an in depth analysis of the Model, Provide feedback if changes of any inputs are needed"

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

def multiplesModel():
    stock_list_pd = pd.read_pickle("StockList")
    target_ticker_string = st.selectbox('Input Target Stock', options=stock_list_pd["symbol"].to_list(), index = 29105)
    comparables_tickers = st.multiselect('Select Comparable Stocks', options=stock_list_pd["symbol"].to_list())
    comparables_tickers.append(target_ticker_string)
    ticker_obj = yf.Ticker(target_ticker_string)
   
    
    st.markdown('# Historical Multiples Vs Average')
    
    all_company_multiples_pe = pd.DataFrame()
    all_company_multiples_ps = pd.DataFrame()
    
    for company in comparables_tickers:
        target_ticker = yf.Ticker(company)

        prices = pd.DataFrame(target_ticker.history(start='2020-01-01')['Close'])
        prices.index = prices.index.date  

        income_stmnt = target_ticker.income_stmt.T 
        income_stmnt.index = income_stmnt.index.date

        # Step 3: Forward-fill missing values in the diluted shares column
        income_stmnt['Diluted Average Shares'].fillna(method='bfill', inplace=True)

        # Step 4: Calculate EPS (Net Income / Diluted Shares)
        income_stmnt['EPS'] = income_stmnt['Net Income'] / income_stmnt['Diluted Average Shares']
        income_stmnt['RPS'] = income_stmnt['Total Revenue'] / income_stmnt['Diluted Average Shares']

        all_dates = pd.date_range(start=min(prices.index.min(), income_stmnt.index.min()),
                            end=max(prices.index.max(), income_stmnt.index.max()), freq='D')
        
        prices = prices.reindex(all_dates)
        #prices['Close'].fillna(method='bfill', inplace=True)
        
        prices = prices.merge(income_stmnt[['EPS']], left_index=True, right_index=True, how='left')
        prices = prices.merge(income_stmnt[['RPS']], left_index=True, right_index=True, how='left')
        
        prices = prices.fillna(method = 'ffill')
        prices['Trailling PE'] = prices['Close']/prices['EPS']
        
        
        prices['Trailling PS'] = prices['Close']/prices['RPS']

        
        all_company_multiples_pe[f'{company}'] = prices['Trailling PE']
        all_company_multiples_ps[f'{company}'] = prices['Trailling PS']
    
    
    all_company_multiples_pe['Average'] = (all_company_multiples_pe.sum(axis=1))/(len(comparables_tickers))
    all_company_multiples_ps['Average'] = (all_company_multiples_ps.sum(axis=1))/(len(comparables_tickers))
    
    with st.form('Plot Historic Multiples'):
        multiple_to_plot = st.selectbox('Select Multiples to Plot', options = ['PE', 'PS'])
        confirm = st.form_submit_button('Plot')
        if confirm:
            if multiple_to_plot == 'PE':
                pe_fig = go.Figure()
                pe_fig.add_trace(go.Scatter(x =all_company_multiples_pe.index, y=all_company_multiples_pe[f'{target_ticker_string}'], name=f'{target_ticker_string} ' ))
                pe_fig.add_trace(go.Scatter(x =all_company_multiples_pe.index, y=all_company_multiples_pe[f'Average'], name=f'Average ' ))
                st.plotly_chart(pe_fig)
            elif multiple_to_plot == 'PS':
                ps_fig = go.Figure()
                ps_fig.add_trace(go.Scatter(x =all_company_multiples_ps.index, y=all_company_multiples_ps[f'{target_ticker_string}'], name=f'{target_ticker_string} ' ))
                ps_fig.add_trace(go.Scatter(x =all_company_multiples_ps.index, y=all_company_multiples_ps[f'Average'], name=f'Average ' )) 
                st.plotly_chart(ps_fig)
    st.markdown("---")
    st.markdown('# Snapshot of Multiples Vs Average')           
    multiples_list = ["Forward PE", "Trailing PE", "EV To EBITDA", "EV To Revenue", "PS"]
    multiples_df = pd.DataFrame()
    multiples_df['Multiple'] = multiples_list
    multiples_df = multiples_df.set_index('Multiple')
    
    comparables_tickers.pop(-1)
    
    
    for company in comparables_tickers:
        ticker = yf.Ticker(company)
        info = ticker.info
        temp_list = []
        fPE = temp_list.append(info.get('forwardPE'))
        tPE = temp_list.append(info.get('trailingPE'))
        evebitda = temp_list.append(info.get('enterpriseToEbitda'))
        evrevenue = temp_list.append(info.get('enterpriseToRevenue'))
        ps = temp_list.append(info.get('currentPrice')/info.get('revenuePerShare'))
        multiples_df[f'{company}'] = temp_list
        temp_list = []
        
    multiples_df['Average'] = (multiples_df.sum(axis=1))/(len(comparables_tickers))
    multiples_df['Median'] = (multiples_df.median(axis=1))
    st.dataframe(multiples_df)
    
    
    multiples_df = multiples_df.T
    
    with st.form('Plot Multiples'):
        multiple_to_plot = st.selectbox('Select Multiple to Plot', options = multiples_list)
        confirm = st.form_submit_button('Plot')
        if confirm:
            multiple_bar_plot = go.Figure()
            multiple_bar_plot.add_trace(go.Bar(x=multiples_df.index, y = multiples_df[multiple_to_plot]))
            st.plotly_chart(multiple_bar_plot)
    st.markdown("---")
    st.markdown('# Valuation')
    st.dataframe(multiples_df)
    
    st.markdown("## Forward PE Valuation")
    
    fpe = (multiples_df['Forward PE'].to_list()[-1])
    feps = (ticker_obj.info.get("forwardEps"))
    st.markdown(f"Forward PE: {fpe}")
    st.markdown(f"Forward EPS: {feps}")
    valuation1 = fpe*feps
    st.markdown(f"#### Valuation: ${round(valuation1,2)}")
    st.markdown("---")
    
    st.markdown("## Trailing PE Valuation")
    
    tpe = (multiples_df['Trailing PE'].to_list()[-1])
    teps = (ticker_obj.info.get("trailingEps"))
    st.markdown(f"Trailing PE: {tpe}")
    st.markdown(f"Trailing EPS: {teps}")
    valuation2 = tpe*teps
    st.markdown(f"#### Valuation: ${round(valuation2,2)}")
    st.markdown("---")
    
    
    st.markdown("## EV/EBITDA Valuation")
    
    evebitda = (multiples_df['EV To EBITDA'].to_list()[-1])
    ebitda = (ticker_obj.info.get("ebitda"))
    st.markdown(f"EV To EBITDA: {evebitda}")
    st.markdown(f"EBITDA: {ebitda}")
    ev = evebitda*ebitda
    st.markdown(f"Enterprise Value: {ev}")
    equityValue = ev-(ticker_obj.info.get("totalDebt")-ticker_obj.info.get("totalCash"))
    st.markdown(f"Equity Value: {equityValue}")
    valuation3 = (equityValue/ticker_obj.info.get("sharesOutstanding"))
    st.markdown(f'#### Valuation: ${round(valuation3,2)}')
    st.markdown("---")
    st.markdown("## EV/Revenue Valuation")
    
    evtorevenue = (multiples_df['EV To Revenue'].to_list()[-1])
    revenue = (ticker_obj.info.get("totalRevenue"))
    st.markdown(f"EV To Revenue: {evtorevenue}")
    st.markdown(f"Revenue: {revenue}")
    ev = evtorevenue*revenue
    st.markdown(f"Enterprise Value: {ev}")
    equityValue = ev-(ticker_obj.info.get("totalDebt")-ticker_obj.info.get("totalCash"))
    st.markdown(f"Equity Value: {equityValue}")
    valuation4 = (equityValue/ticker_obj.info.get("sharesOutstanding"))
    st.markdown(f'#### Valuation: ${round(valuation4,2)}')
    st.markdown("---")
    st.markdown(f"### Average of All Valuations:$ {round(((valuation1+valuation2+valuation3+valuation4)/(4)),2)}")
    st.markdown(f'### Current Price: ${ticker_obj.info.get('currentPrice')}')
    
    st.markdown("# Summary")
     
    user_prompt = f'''You are a financial analyst and are analyzing a company with the ticker symbol {target_ticker_string}.
    You are using a Multiples Valuation model.
    This is the list of comparable companies {comparables_tickers} you have used. 
    Here are all the multiples of the comparable Companies {multiples_df.to_string()}. 
    Here is the implied value of the stock using The Forward PE Multiple: {round(valuation1,2)}.
    Here is the implied value of the stock using The Trailing PE Multiple: {round(valuation2,2)}.
    Here is the implied value of the stock using The EV to Revenue Multiple: {round(valuation3,2)}.
    here is the implied value of the stock using The EV to EBITDA Multiple: {round(valuation4,2)}.
    Here is the Actual current price of the stock: {ticker_obj.info.get('currentPrice')}
    Using this data provide a detailed explanation of whether the Company is fairly valued or not.
    Also give any crticics of the model '''

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

def k_means_clustering():
    st.markdown("# KMeans Clustering")
    multiples_df = pd.read_csv('Multiples_Database.csv')
    multiples_df = multiples_df.drop(columns = ['Unnamed: 0'])
    k_inputted = st.number_input("Input Number of Clusters", min_value=2, step=1)
    target_stock = st.selectbox("Enter Stock",options=multiples_df['Company'].to_list())
    metrics = multiples_df.columns.to_list()
    metrics.remove('Company')
    metrics.remove('Sector')
    metrics.remove('Name')
    metrics.remove('Industry')
    multiples_to_cluster_on = st.multiselect("Select Metrics to Cluster on", options=metrics)
    
        
    if st.button("Cluster") :
        df_no_nan = multiples_df.dropna()
        df_no_nan = df_no_nan.reset_index(drop=True)
        df_numeric = df_no_nan.drop(columns=["Company", "Industry", "Sector", "Name"])
        metrics_to_drop = list(filter(lambda item: item not in multiples_to_cluster_on, df_numeric.columns.to_list()))
        df_numeric = df_numeric.drop(columns=metrics_to_drop)
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(df_numeric)
        df_scaled = pd.DataFrame(np_scaled,columns=df_numeric.columns)
        df_scaled.insert(0, "Industry", df_no_nan["Industry"])
        df_scaled.insert(0, "Sector", df_no_nan["Sector"])
        df_scaled.insert(0, "Name", df_no_nan["Name"])
        df_scaled.insert(0, "Company", df_no_nan["Company"])
        for feature in df_scaled.columns.to_list()[4:]: 
            df_scaled_no_outliers = df_scaled.drop(df_scaled[df_scaled[feature] > 35].index)
        df_scaled_no_outliers = df_scaled_no_outliers.reset_index(drop=True)
        df_scaled_no_outliers_numeric = df_scaled_no_outliers.drop(columns=["Company", "Industry", "Sector", "Name"])
        inertia_list = []
        for k in range(3,100):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(df_scaled_no_outliers_numeric)
            inertia_list.append(kmeans.inertia_)
        inertia_df = pd.DataFrame(inertia_list)
        inertia_fig = go.Figure()
        inertia_fig.add_trace(go.Scatter(x = inertia_df.index, y =inertia_df[0] ))
        st.markdown("## Elbow Chart")
        st.plotly_chart(inertia_fig)
            
        labels = KMeans(n_clusters=k_inputted).fit_predict(df_scaled_no_outliers_numeric)
        df_scaled_no_outliers_numeric["Cluster"] = labels
        df_scaled_no_outliers_numeric.insert(0,"Sector", df_scaled_no_outliers["Sector"])
        df_scaled_no_outliers_numeric.insert(0,"Industry", df_scaled_no_outliers["Industry"])
        df_scaled_no_outliers_numeric.insert(0,"Name", df_scaled_no_outliers["Name"])
        df_scaled_no_outliers_numeric.insert(0,"Company", df_scaled_no_outliers["Company"])
        df_final = df_scaled_no_outliers_numeric
        st.dataframe(df_final)
        target_cluster = df_final[df_final["Company"] == target_stock]["Cluster"].values[0]
        target_sector = df_final[df_final["Company"] == target_stock]["Sector"].values[0]
        target_industry = df_final[df_final["Company"] == target_stock]["Industry"].values[0]
        comparable_Companies_list = df_final[df_final["Cluster"]==target_cluster]["Company"].to_list()
        comparable_Sectors_list = df_final[df_final["Cluster"]==target_cluster]["Sector"].to_list()
        comparable_Industries_list = df_final[df_final["Cluster"]==target_cluster]["Industry"].to_list()
        comparable_Names_list = df_final[df_final["Cluster"]==target_cluster]["Name"].to_list()
        comparables_df = pd.DataFrame()
        comparables_df["Company"] = comparable_Companies_list
        comparables_df["Name"] = comparable_Names_list
        comparables_df["Sector"] = comparable_Sectors_list
        comparables_df["Industry"] = comparable_Industries_list
        st.markdown("## Same Cluster")
        st.dataframe(comparables_df[comparables_df["Sector"] == target_sector])
        st.markdown("## Same Cluster and Sector")
        st.dataframe(comparables_df[comparables_df["Sector"] == target_sector])
        st.markdown("## Same Cluster and Sector and Industry")
        st.dataframe(comparables_df[comparables_df["Industry"] == target_industry])
        st.markdown("## Same Sector")
        st.dataframe(df_no_nan[df_no_nan['Sector'] == target_sector].loc[:,['Company', 'Name', 'Sector','Industry' ]])
        st.markdown("## Same Industry")
        st.dataframe(df_no_nan[df_no_nan['Industry'] == target_industry].loc[:,['Company', 'Name', 'Sector','Industry' ]])
        st.markdown("## Cluster Visualisation")
        if len(multiples_to_cluster_on) == 1:
            st.markdown("Not Enough Metrics to Cluster on")
        elif len(multiples_to_cluster_on) == 2:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_result = tsne.fit_transform(df_scaled_no_outliers_numeric.drop(columns=["Company", "Industry", "Sector", "Name", "Cluster"]))
            df_tsne = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2"])
            df_tsne['Cluster'] = df_final["Cluster"]
            df_tsne['Company'] = df_final["Company"]
            tsne_fig = go.Figure()
            for cluster in df_tsne["Cluster"].unique():
                
                cluster_data = df_tsne[df_tsne["Cluster"] == cluster]
                tsne_fig.add_trace(go.Scatter(x = cluster_data['tSNE1'], y = cluster_data['tSNE2'] , name=f"Cluster: {cluster}", mode = 'markers',), )
                tsne_fig.update_traces(marker={'size': 3,})
                tsne_fig.update_layout(
                autosize=False,
                width=900,
                height=600,
            )
            st.plotly_chart(tsne_fig)
        else:
            tsne = TSNE(n_components=3, perplexity=30, random_state=42)
            tsne_result = tsne.fit_transform(df_scaled_no_outliers_numeric.drop(columns=["Company", "Industry", "Sector", "Name", "Cluster"]))
            df_tsne = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2", 'tSNE3'])
            df_tsne['Cluster'] = df_final["Cluster"]
            df_tsne['Company'] = df_final["Company"]
            tsne_fig = go.Figure()
            for cluster in df_tsne["Cluster"].unique():
                
                cluster_data = df_tsne[df_tsne["Cluster"] == cluster]
                tsne_fig.add_trace(go.Scatter3d(x = cluster_data['tSNE1'], y = cluster_data['tSNE2'],z =cluster_data['tSNE3'] , name=f"Cluster: {cluster}", mode = 'markers',), )
                tsne_fig.update_traces(marker={'size': 3,})
                tsne_fig.update_layout(
                autosize=False,
                width=900,
                height=600,
            )
            st.plotly_chart(tsne_fig)

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
    df = pd.read_csv("Multiples_Database.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    df = df.dropna()
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
    
    
    
    
    st.markdown("## Sector Performance")
    lookback_period = st.number_input("Enter Lookback Period", min_value=2, max_value=1257, step=10)
    sector_proxies = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
    sector_proxies_names = ['Materials', 'Communications', 'Energy', 'Financials', 'Industrials', 'Technology', 'Consumer Staples', 'Real Estate', 'Utilities', 'Healthcare', 'Consumer Discretionary']
    hm = yf.Tickers(sector_proxies)
    df = hm.history(period='5y', interval='1d',auto_adjust=False)['Close']
    performance_df = ((df.iloc[-1,:]-df.iloc[-lookback_period,:])/df.iloc[-lookback_period,:])*100
    performance_df = pd.DataFrame(performance_df, columns=["Value", ])
    performance_df["Name"] = sector_proxies_names
    performance_fig = go.Figure()
    temp = performance_df.sort_values(['Value'])
    performance_fig.add_trace(go.Bar(x = temp['Value'], y = temp["Name"], orientation='h'))
    performance_fig.update_xaxes(title_text = f"Performance over {lookback_period-1} Days")
    performance_fig.update_yaxes(title_text = "Sector")
    performance_fig.update_layout(
                autosize=False,
                width=1000,
                height=500,
            )
    st.plotly_chart(performance_fig)
    
    
    st.markdown("-----")
    st.markdown("## Headlines")
    i=0
    for ticker in sector_proxies:
        st.markdown(f"### {sector_proxies_names[i]}")
        oi = yf.Ticker(ticker)
        news = oi.news
        for article in news[0:3]:
            st.markdown(f"#### {article["title"]}")
            st.markdown(article['relatedTickers'])
            st.markdown(article['link'])
            
        st.markdown("-----")
        i = i+1
    
     
with st.sidebar:
    selected = option_menu(
        menu_title = 'Models',
        options = [ 'DCF Model', 'Multiples Model', 'K Means Clustering', 'Sector Screener','Portfolio Variance Calculator'],
        orientation='vertical',
        icons = ['house', 'buildings', 'lock', 'buildings','buildings' ])


    
if selected == 'DCF Model':
    dcfModel()

if selected == 'Multiples Model':
    multiplesModel()
    
if selected == 'K Means Clustering':
    k_means_clustering()

if selected == 'Sector Screener':
    sector_screener()
    
if selected == 'Portfolio Variance Calculator':
    risk_analysis()





        