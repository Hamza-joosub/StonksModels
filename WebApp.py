from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly_express  as px
import plotly.graph_objects as go
import requests
from streamlit_option_menu import option_menu
import math
import random
from dateutil.relativedelta import relativedelta

#db = pd.read_pickle('us_stock_database.pkl')

st.set_page_config(layout="wide")

key = "6ulfs8VItWZcKZTMzNJxwmikpQvSF1cI"

def expand_numbers(df, list_of_columns):
    for name_of_column in list_of_columns:
        most_recent_value = 0
        for row in df.index:
            if pd.isna(df.loc[row, name_of_column]) and most_recent_value == 0:
                continue
            elif pd.isna(df.loc[row, name_of_column]):
                df.loc[row, name_of_column] = most_recent_value
            else:
                most_recent_value = df.loc[row, name_of_column]
    return df

def unpack_quarterly_data(ticker,databaseI ):
    raw_data = databaseI.query(f"Ticker == '{ticker}'")
    last_annual_date = raw_data.loc[0,'Date(FQ)']
    raw_data = raw_data.drop(['Ticker', 'Date(FQ)', 'Date(FY)'], axis=1)
    raw_data = raw_data.transpose()
    raw_data = raw_data.reset_index()
    raw_data.rename(columns = {'index':'feature', 0 : 'value'}, inplace=True)
    raw_data.dropna(inplace=True)
    annual_df = raw_data.query("feature.str.contains('FQ')")
    annual_stock_data = pd.DataFrame()
    current = annual_df.query(f'feature.str.endswith(")")')
    current = current.reset_index(drop=True)
    annual_stock_data["feature"] = current['feature']
    annual_stock_data['Current'] = current['value']
    for year in range(1,9):
        temp = annual_df.query(f'feature.str.endswith("{year}")')
        temp.reset_index(inplace=True, drop=True)
        annual_stock_data[f"-{year}"] = temp[f"value"]
    annual_stock_data.set_index('feature', inplace=True)
    annual_stock_data.rename(columns = {'Current' : '0'}, inplace=True)
    annual_stock_data = annual_stock_data.transpose()
    annual_stock_data = annual_stock_data.reindex(index=annual_stock_data.index[::-1])
    annual_stock_data_features_list  = annual_stock_data.columns.to_list()
    date_list = []
    for year in annual_stock_data.index.to_list():
        yearsback = abs(int(year))
        date = pd.to_datetime(last_annual_date) - relativedelta(years=yearsback)
        date_list.append(date)
    annual_stock_data.insert(loc=0, column='Date', value=date_list)
    annual_stock_data = annual_stock_data.reset_index(drop=True)
    return annual_stock_data, annual_stock_data_features_list

def unpack_annual_data(tickernameI, dbI):
    raw_data = dbI.query(f"Ticker == '{tickernameI}'")
    last_annual_date = raw_data.loc[0,'Date(FY)']
    raw_data = raw_data.drop(['Ticker', 'Date(FQ)', 'Date(FY)'], axis=1)
    raw_data = raw_data.transpose()
    raw_data = raw_data.reset_index()
    raw_data.rename(columns = {'index':'feature', 0 : 'value'}, inplace=True)
    raw_data.dropna(inplace=True)
    annual_df = raw_data.query("feature.str.contains('FY')")
    annual_stock_data = pd.DataFrame()
    current = annual_df.query(f'feature.str.endswith(")")')
    current = current.reset_index(drop=True)
    annual_stock_data["feature"] = current['feature']
    annual_stock_data['Current'] = current['value']
    for year in range(1,9):
        temp = annual_df.query(f'feature.str.endswith("{year}")')
        temp.reset_index(inplace=True, drop=True)
        annual_stock_data[f"-{year}"] = temp[f"value"]
    annual_stock_data.set_index('feature', inplace=True)
    annual_stock_data.rename(columns = {'Current' : '0'}, inplace=True)
    annual_stock_data = annual_stock_data.transpose()
    annual_stock_data = annual_stock_data.reindex(index=annual_stock_data.index[::-1])
    annual_stock_data_features_list  = annual_stock_data.columns.to_list()
    date_list = []
    for year in annual_stock_data.index.to_list():
        yearsback = abs(int(year))
        date = pd.to_datetime(last_annual_date) - relativedelta(years=yearsback)
        date_list.append(date)
    annual_stock_data.insert(loc=0, column='Date', value=date_list)
    annual_stock_data = annual_stock_data.reset_index(drop=True)
    return annual_stock_data, annual_stock_data_features_list

def get_stock_price_db(tickernameI, is_dfI,years):
    is_dfL = is_dfI.iloc[-years:]
    is_dfL = is_dfL.reset_index(drop=True)
    tickernameL = tickernameI
    start_dateL = is_dfL["Date"][0].date().strftime("%Y-%m-%d")
    rows = is_dfL.shape[0] -1
    start_data_offset1 = (is_dfL["Date"][0] + pd.Timedelta(days = 1)).date().strftime("%Y-%m-%d")
    start_data_offset2 = (is_dfL["Date"][0] + pd.Timedelta(days = 2)).date().strftime("%Y-%m-%d")
    start_data_offset3 = (is_dfL["Date"][0] + pd.Timedelta(days = 3)).date().strftime("%Y-%m-%d")
    start_data_offset4 = (is_dfL["Date"][0] + pd.Timedelta(days = 4)).date().strftime("%Y-%m-%d")
    end_date = is_dfL["Date"][rows].date().strftime("%Y-%m-%d")
    end_date_offset1 = (is_dfL["Date"][rows] + pd.Timedelta(days = 1)).date().strftime("%Y-%m-%d")
    end_date_offset2 = (is_dfL["Date"][rows] + pd.Timedelta(days = 2)).date().strftime("%Y-%m-%d")
    end_date_offset3 = (is_dfL["Date"][rows] + pd.Timedelta(days = 3)).date().strftime("%Y-%m-%d")
    end_date_offset4 = (is_dfL["Date"][rows] + pd.Timedelta(days = 4)).date().strftime("%Y-%m-%d")
    now = datetime.today().strftime('%Y-%m-%d')
    price_dfL = pd.DataFrame(yf.download(tickers=tickernameL,start = "2017-01-01" , end =now, interval="1d")["Close"])
    price_dfL.reset_index(inplace=True)
    if(price_dfL[price_dfL["Date"] == start_dateL].count()[0] == 0): #no price for start date use first price
        start_priceL = price_dfL.iloc[0,1]
    else:
        start_priceL = price_dfL.query(f"Date == '{start_dateL}' | Date == '{start_data_offset1}' | Date == '{start_data_offset2}'| Date == '{start_data_offset3}'| Date == '{start_data_offset4}' ").iloc[0,1]
    end_price = price_dfL.query(f"Date == '{end_date}' | Date == '{end_date_offset1}' | Date == '{end_date_offset2}' | Date == '{end_date_offset3}' | Date == '{end_date_offset4}' ").iloc[0,1]
    price_cagrL = round(((end_price/start_priceL)**(1/is_dfL.shape[0])-1)*100,2)
    return price_dfL, price_cagrL, start_dateL

def get_description_data(tickernameI):
    tickernameL = tickernameI
    description_url = f"https://financialmodelingprep.com/api/v3/profile/{tickernameL}?apikey={key}"
    description_dfL = pd.DataFrame(requests.get(description_url).json())
    description_features_listL = description_dfL.columns.values.tolist()
    return description_dfL, description_features_listL

def get_rev_seg_data(tickernameI, periodI):
    tickernameL = tickernameI
    periodL = periodI
    url = f'https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={tickernameL}&structure=flat&period={periodL}&apikey=6ulfs8VItWZcKZTMzNJxwmikpQvSF1cI'
    rev_seg_json = requests.get(url).json()
    rev_seg_dfL = pd.DataFrame()
    for date in rev_seg_json:
        temp_df = pd.DataFrame(date).transpose()
        rev_seg_dfL = pd.concat([rev_seg_dfL, temp_df])
    last_features = rev_seg_dfL.iloc[0,:]
    last_features = last_features.dropna()
    last_features_list = last_features.index.to_list()
    rev_seg_dfL = rev_seg_dfL[last_features_list]
    rev_seg_dfL = rev_seg_dfL.dropna()
    return rev_seg_dfL

def process_company_description(description_dfI):
    descriptionL = description_dfI.iloc[0,17]
    return descriptionL

def process_revenue_metrics_db(annual_stock_dataI, years, periodI):
    is_dfL = annual_stock_dataI
    
    is_dfL = annual_stock_dataI.iloc[-years:]
    is_dfL.reset_index(inplace = True, drop = True)
    is_dfL = is_dfL.drop(['Date'], axis = 1)
    is_df_features_listL =  is_dfL.columns.to_list()
    rows = is_dfL.shape[0]-1
    num_of_periods = is_dfL.shape[0]
    period = periodI

    cagr_list = []
    total_growth_list = []
    percent_of_revenue_list = []
    if period == 'annual':
        period_name = 'FY'
    if period == 'quarterly':
        period_name = 'FQ'
    for i in range(0,len(is_df_features_listL)):
        #CAGR
        
        end = is_dfL.iloc[rows,i]
        start = is_dfL.iloc[0,i]
        
        if start == 0:
            cagr = float("nan")
            cagr_list.append(cagr)
        else:
            cagr = ((end/start)**(1/num_of_periods)-1).real
            cagr_list.append(round(cagr*100,2))
        #Total Growth
        if start == 0:
            total_growth = float("nan")
            total_growth_list.append(total_growth)
        else:
            total_growth = round(((end-start)/start)*100,2)
            total_growth_list.append(total_growth)
        #Percent of Revenue
        revenue = is_dfL.loc[rows,f"revenue({period_name})"]
        percent_of_revenue = abs(round(((end/revenue)*100),2))
        percent_of_revenue_list.append(percent_of_revenue)
    id_df_summary_dfL = pd.DataFrame()
    id_df_summary_dfL["Feature"] = is_df_features_listL
    id_df_summary_dfL[f"{period_name} Compounded Growth Rate %"] = cagr_list
    id_df_summary_dfL["Total Growth %"] = total_growth_list
    id_df_summary_dfL["Percentage Of Revenue(Last)"] = percent_of_revenue_list
    return id_df_summary_dfL, is_dfL

def plot_stock_price(price_dfI, price_cagrI):
    price_dfL= price_dfI
    price_cagrL = price_cagrI
    price_dfL["Date"] = pd.to_datetime(price_dfL["Date"])
    
    price_plot = px.line(x = price_dfL["Date"],y = price_dfL["Close"], title=(f"CAGR: {price_cagrL} %"))
    
    return price_plot

def plot_description(descriptionI):
    descriptionL = descriptionI
    return descriptionL

def plot_revenue_metrics_db(id_df_summary_dfI,periodI):
    id_df_summary_dfL = id_df_summary_dfI
    periodL = periodI
    if periodL == 'annual':
        period_abbreviation = 'FY'
    else:
        period_abbreviation = 'FQ'
    metric_choices = [f"{period_abbreviation} Compounded Growth Rate %", "Total Growth %", "Percentage Of Revenue(Last)"]
    features_list = id_df_summary_dfL["Feature"].unique().tolist()
    with st.form("my_form"):
        metric = st.selectbox("Choose Metric", options=metric_choices)
        features_chosen = st.multiselect("Choose Feature", options=features_list)
        rev_metr_submit = st.form_submit_button("Confirm")
    
    if rev_metr_submit | st.session_state['revenue_metrics'] == True:
        st.session_state['revenue_metrics'] = True
        cagr_plot = go.Figure(go.Bar(x = features_chosen, y =id_df_summary_dfL[metric] ))
        
        
        return cagr_plot

def plot_historic_multiples(price_dfI):
    price_dfL = price_dfI
    with st.form("my_form2"):
        
        mutliple_to_plot = st.multiselect("Select Multiple to Plot",options = price_dfL.columns)
        hist_mult_submit = st.form_submit_button("Confirm")
            
    if hist_mult_submit | st.session_state['historic_multiples'] == True:
        st.session_state['historic_multiples'] = True
        multiples_plot = go.Figure()
        for feature in mutliple_to_plot:
            mean = price_dfL[feature].mean()
            std = price_dfL[feature].std()
            multiples_plot.add_hline(y = mean, line_width = 3, line_dash = "dash", line_color = "green", name = "mean")
            multiples_plot.add_hline(y = mean + std, line_width = 1, line_dash = "dash", line_color = "orange", name = "+1 Std")
            multiples_plot.add_hline(y = mean - std, line_width = 1, line_dash = "dash", line_color = "orange", name = "-1 Std")
            multiples_plot.add_trace(go.Scatter(x = price_dfL['Date'],y = price_dfL[feature]))
        return multiples_plot

def process_historic_multiples_db(price_dfI, is_dfI, years, periodI ):
    price_dfL = price_dfI
    is_dfL = is_dfI.iloc[-years:]
    if periodI == 'annual':
        period_name = "FY"
    else:
        period_name = "FQ"
    Multiples_dfL = pd.DataFrame()
    Multiples_dfL["Date"] = is_dfL["Date"]
    Multiples_dfL["revenue"] = is_dfL[f"revenue({period_name})"]
    Multiples_dfL["weightedAverageShsOutDil"] = is_dfL[f"weightedAverageShsOutDil({period_name})"]
    Multiples_dfL["Basic RPS"] = Multiples_dfL["revenue"]/Multiples_dfL["weightedAverageShsOutDil"]
    Multiples_dfL["epsdiluted"] = is_dfL[f'epsdiluted({period_name})']
    price_dfL["EPS"] = float("nan")
    price_dfL["RPS"] = float("nan")
    for i in Multiples_dfL.iterrows():
        rps = i[1][3]
        eps = i[1][4]
        date = i[1][0]
        dateoffset1 = date + pd.Timedelta(days = 1)
        dateoffset2 = date + pd.Timedelta(days = 2)
        dateoffset3 = date + pd.Timedelta(days = 3)
        dateoffset4 = date + pd.Timedelta(days = 4)
        price_dfL.loc[price_dfL.query(f"Date == '{date}' | Date == '{dateoffset1}' | Date == '{dateoffset2}'| Date == '{dateoffset3}' | Date == '{dateoffset4}'").index,"EPS"] = eps
        price_dfL.loc[price_dfL.query(f"Date == '{date}' | Date == '{dateoffset1}' | Date == '{dateoffset2}'| Date == '{dateoffset3}' | Date == '{dateoffset4}'").index,"RPS"] = rps
    price_features_listL = price_dfL.columns.values.tolist()
    price_dfL = expand_numbers(price_dfL, price_features_listL)
    price_dfL["PE"] = price_dfL["Close"]/price_dfL["EPS"]
    price_dfL["PS"] = price_dfL["Close"]/price_dfL["RPS"]
    price_dfL["Volatility"] = price_dfL["Close"].pct_change()
    return price_dfL, Multiples_dfL

def plot_free_graph1( is_dfI,is_df_features_listI, fp_dfI,fp_df_feature_listI,  cf_dfI,cf_df_feature_listI,years  ):
    is_df_features_listL = is_df_features_listI
    fp_df_feature_listL = fp_df_feature_listI
    cf_df_feature_listL = cf_df_feature_listI
    allFeatures_listL = is_df_features_listL + fp_df_feature_listL + cf_df_feature_listL
    is_dfL = is_dfI.iloc[-years:]
    fp_dfL = fp_dfI.iloc[-years:]
    cf_dfL = cf_dfI.iloc[-years:]
    allFeatures_listL = is_df_features_listL + fp_df_feature_listL + cf_df_feature_listL
    #randomKey = random.randint(0,200)
    with st.form(f"Free Graph Form"):
        
        to_plot1 = st.multiselect("Things to Plot",options = allFeatures_listL)
        freegraph1_submit = st.form_submit_button("confirm")
    if freegraph1_submit | st.session_state["freegraph1"] == True:
        st.session_state["freegraph1"] = True
        free_plot1 = go.Figure()
        for metric in to_plot1:
            if metric in is_df_features_listL:
                free_plot1.add_trace(go.Scatter(x = is_dfL["Date"], y = is_dfL[metric], name = f"{metric}"))
                continue
            if metric in fp_df_feature_listL:
                free_plot1.add_trace(go.Scatter(x = fp_dfL["Date"], y = fp_dfL[metric], name = f"{metric}"))
                continue
            if metric in cf_df_feature_listL:
                free_plot1.add_trace(go.Scatter(x = cf_dfL["Date"], y = cf_dfL[metric], name = f"{metric}"))
                continue
        return free_plot1

def plot_rev_segmentation_snapshot(rev_seg_dfI):
    rev_seg_dfL = rev_seg_dfI.transpose()
    fig =  px.pie(names=rev_seg_dfL.index.to_list(),values = rev_seg_dfL.iloc[:,0], color_discrete_sequence= px.colors.sequential.deep )
    return fig

def plot_rev_seg_over_time(rev_seg_dfI):
    rev_seg_dfL = rev_seg_dfI
    rev_seg_dfL.reset_index(inplace=True)
    rev_seg_dfL.rename(columns = {'index' : 'date'}, inplace=True)
    rev_seg_features_listL = rev_seg_dfL.columns.to_list()
    rev_seg_over_time = go.Figure()
    for feature in rev_seg_features_listL[1:]:
        rev_seg_over_time.add_scatter(x = rev_seg_dfL['date'], y = rev_seg_dfL[feature],name =feature,   )
    return rev_seg_over_time

def screener():
    screener_raw = pd.read_pickle('sample_db.pkl')
    screener_raw = screener_raw.drop(['Date(FQ)','Date(FY)'], axis = 1)
    st.table(screener_raw)
        
def Calc_correlation_Matrix(Tickers, Start, End, Interval):
    portfolio_tickers = Tickers
    start = Start
    end = End
    rawData = yf.download(tickers=portfolio_tickers,start=start, end =end, interval =Interval )["Close"]  
    diffirenced_data = rawData.pct_change()
    corr_matrix = diffirenced_data.corr()
    fig = px.imshow(corr_matrix,aspect = 'auto', color_continuous_scale='sunsetdark',)
    
    return fig

def portfolio_variance(Tickers, Start, End, Interval):
    st.markdown("# Porfolio Variance")
    weights = []
    portfolio_tickers = Tickers
    with st.form("form4"):
        for i in range(len(Tickers)):
            temp_weight = st.number_input(f'weight: {Tickers[i]}', key=i,step = 5 )
            weights.append(temp_weight)
        variance_weights = st.form_submit_button("confirm")   
    if variance_weights:    
        start = Start
        end = End
        rawData = yf.download(tickers=portfolio_tickers,start=start, end =end, interval =Interval )["Close"]  
        covariance_Matrix = rawData.corr()
        total_risk_Exposure = 0
    
        tickerNumber = 0
        for ticker in portfolio_tickers:
            st.markdown(f"Volatility of {ticker}: {round(((rawData[ticker].pct_change()).std()*100),2)}%")
            total_risk_Exposure = total_risk_Exposure + ((rawData[ticker].pct_change().std())**2)*((weights[tickerNumber]/100)**2)
            tickerNumber = tickerNumber+1
        
        for ticker in range(len(portfolio_tickers)-1):
            for ticker2 in range(ticker, len(portfolio_tickers)):
                if ticker == ticker2:
                    continue
                total_risk_Exposure = total_risk_Exposure + 2*covariance_Matrix.iloc[ticker,ticker2]*(weights[ticker]/100)*(weights[ticker2]/100)
        
    
        portfolio_Variance = f"{round(total_risk_Exposure*100,2)}%"
        Portfolio_Volatility = f"{round(math.sqrt(abs(total_risk_Exposure*100)),2)}%"
        st.markdown(f'### Porfolio Variance: {Portfolio_Volatility}')

def risk_analysis():
    stock_list_pd = pd.read_pickle("StockList")
    portfolio_stocks = st.multiselect("Choose Stocks", options=stock_list_pd["symbol"].to_list())
    if len(portfolio_stocks)>1:
        st.markdown("# Portfolio Correlation")
        st.plotly_chart(Calc_correlation_Matrix(Tickers = portfolio_stocks, Start = "2022-01-01" , End = datetime.today().strftime('%Y-%m-%d'), Interval = "1d"))
        portfolio_variance(Tickers = portfolio_stocks, Start = "2022-01-01" , End = datetime.today().strftime('%Y-%m-%d'), Interval = "1d")

def Stock_Analysis():
    if 'revenue_metrics' not in st.session_state:
        st.session_state.revenue_metrics = False
    if 'historic_multiples' not in st.session_state:
        st.session_state.historic_multiples = False
    if 'freegraph1' not in st.session_state:
        st.session_state.freegraph1 = False
    if 'stock_submission' not in st.session_state:
        st.session_state.stock_submission = False
    db = pd.read_pickle('sample_db.pkl')
    stock_list_pd = pd.read_pickle("StockList")
    st.markdown('# Enter Stock Name')
    
    tickername = st.selectbox("Input Stock Ticker", options=stock_list_pd["symbol"].to_list(), index = stock_list_pd["symbol"].to_list().index("AAPL") )
    period = st.selectbox("Select Period", options=["annual", "quarterly"])
    years_of_data = st.number_input("Number of Reports", step =1, min_value = 2, )
    
    if period == 'annual':
        stock_data, annual_stock_data_features_list = unpack_annual_data(tickername, db)
    else:
        stock_data, annual_stock_data_features_list = unpack_quarterly_data(tickername, db)
        
    price_df, price_cagr, start_date = get_stock_price_db(tickername, stock_data, years_of_data)
    description_df, description_features_list = get_description_data(tickername)
    s = process_company_description(description_df )
    price_df, Multiples_df = process_historic_multiples_db(price_df, stock_data, years_of_data, period)
    annual_stock_data_summary, stock_data = process_revenue_metrics_db(stock_data,years_of_data, period)
    
    with st.expander("Price"):
        st.markdown('# Price')
        st.plotly_chart(plot_stock_price(price_df, price_cagr), use_container_width=True)
        
    with st.expander("Description"):
        st.markdown(plot_description(s))
        st.markdown('---')
    
    with st.expander("Profitability Metrics"):
        st.plotly_chart(plot_revenue_metrics_db(annual_stock_data_summary, period))
        
    with st.expander("Historic Multiples"):
        st.plotly_chart(plot_historic_multiples(price_df ))
    
    usa_stock_list = stock_list_pd.query("exchangeShortName == 'NYSE' | exchangeShortName == 'NASDAQ'")['symbol'].to_list()
    if tickername in usa_stock_list:
        with st.expander("Revenue Breakdown"):
            st.markdown("# Revenue Breakdown")
            rev_seg_df = get_rev_seg_data(tickername,period )
            
            st.plotly_chart(plot_rev_segmentation_snapshot(rev_seg_df), use_container_width=True)
            st.plotly_chart(plot_rev_seg_over_time(rev_seg_df), use_container_width=True)
            st.markdown('---')
            
def topbar():
    last_day = (datetime.today() - pd.Timedelta(days=4)).strftime('%Y-%m-%d')
    now = datetime.today().strftime('%Y-%m-%d')
    indicies_list = ["QQQ", "SPY", "DIA"]
    col1,col2,col3 = st.columns(3)
    column_list = [col1,col2,col3]
    column_chooser= 0
    indices = pd.DataFrame(yf.download(tickers=indicies_list, start=last_day, end = now, interval="1d" )["Close"])
    for indice in indicies_list:
        change = (round((indices[indice][-1] - indices[indice][-2])/indices[indice][-1]*100,2))
        column_list[column_chooser].metric(label = f"{indice}", value = int(indices[indice][-1]), delta=change)
        column_chooser = column_chooser + 1

topbar()

selected = option_menu(
        menu_title = None,
        options = ['Screener','Stock Analysis(API)', 'Risk'],
        orientation='vertical',
        icons = ['house', 'buildings', 'lock'])

if selected == 'Screener':
    screener()
if selected == 'Stock Analysis(API)':
    Stock_Analysis()
if selected == 'Risk':
    risk_analysis()